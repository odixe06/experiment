import unittest
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import numpy as np
import torch
from torch.nn import functional as F
from core.central_backbones import build_model, classification_loss, predict


class CentralModelsTest(unittest.TestCase):
    def test_driver_and_epoch_resume(self):
        from scripts import run_central_backbones as driver
        from scripts.audit_central_backbones import audit
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory(dir=".") as work:
            root = Path(work)
            rng = np.random.default_rng(1)
            (root/"identity.json").write_text('{"fixture":1}')
            for split,n in (("train",13),("test",7)):
                np.save(root/f"{split}_X.npy", rng.normal(size=(n,77)).astype(np.float16))
                np.save(root/f"{split}_Y.npy", rng.integers(0,16,size=n,dtype=np.int8))
            for name in ("mlp","tabm","bert","sae","sae-ce","cnn-bigru-attention"):
                out = root/name
                args = ["train","--model",name,"--out",str(out),"--cache",str(root),"--cpu",
                        "--batch","8","--micro","3","--epochs","2"]
                def launch(arguments):
                    with patch.object(sys,"argv",arguments):
                        driver.main()
                launch(args+["--stop-after","1"])
                launch(args+["--resume"])
                final = out/"epoch_002"
                self.assertEqual(audit(out), 2)
                pred = np.load(final/"pred_0.npy")
                true = np.load(out/"epoch_001/truth_0.npy").astype(np.int64)
                cm = np.bincount(16*true+pred,minlength=256).reshape(16,16)
                np.testing.assert_array_equal(cm,np.load(final/"cm.npy"))
                row = json.loads((final/"metrics.json").read_text())
                self.assertEqual((row["train_rows"],row["eval_rows"],row["applied"]),(13,7,2))
                if name in ("tabm","sae","cnn-bigru-attention"):
                    continuous = root/f"continuous-{name}"
                    launch([str(continuous) if v == str(out) else v for v in args])
                    expected = torch.load(continuous/"epoch_002/weights.pt",weights_only=True)
                    for key,tensor in torch.load(final/"weights.pt",weights_only=True).items():
                        self.assertTrue(torch.equal(expected[key],tensor))

    def test_seed_and_sae_ce(self):
        """--seed changes init/streams and is recorded; sae-ce is the FL SAE network (CE-only objective)."""
        from scripts import run_central_backbones as driver
        from core.sae import SupervisedSAE, fl_config
        ce, fl = build_model("sae-ce"), SupervisedSAE(fl_config())
        self.assertEqual({k: v.shape for k, v in ce.state_dict().items()}, {k: v.shape for k, v in fl.state_dict().items()})
        self.assertEqual(sum(p.numel() for p in ce.parameters()), 7613)
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory(dir=".") as work:
            root = Path(work)
            rng = np.random.default_rng(1)
            (root/"identity.json").write_text('{"fixture":1}')
            for split,n in (("train",13),("test",7)):
                np.save(root/f"{split}_X.npy", rng.normal(size=(n,77)).astype(np.float16))
                np.save(root/f"{split}_Y.npy", rng.integers(0,16,size=n,dtype=np.int8))
            weights = {}
            for seed in ("42", "41"):
                out = root/f"mlp-{seed}"
                with patch.object(sys,"argv",["train","--model","mlp","--out",str(out),"--cache",str(root),"--cpu",
                                              "--batch","8","--micro","3","--epochs","1","--seed",seed]):
                    driver.main()
                self.assertEqual(json.loads((out/"config.json").read_text())["seed"], int(seed))
                weights[seed] = torch.load(out/"epoch_001/weights.pt",weights_only=True)
            self.assertFalse(all(torch.equal(weights["42"][k], weights["41"][k]) for k in weights["42"]))

    def test_two_rank_ddp(self):
        """world = 2 (gloo, CPU): DDP unused-parameter check (sae-ce decoder) and the config.json write/read order."""
        import os, shutil, subprocess
        if shutil.which("torchrun") is None:
            self.skipTest("torchrun not on PATH")
        with tempfile.TemporaryDirectory(dir=".") as work:
            root = Path(work)
            rng = np.random.default_rng(1)
            (root/"identity.json").write_text('{"fixture":1}')
            for split,n in (("train",29),("test",9)):
                np.save(root/f"{split}_X.npy", rng.normal(size=(n,77)).astype(np.float16))
                np.save(root/f"{split}_Y.npy", rng.integers(0,16,size=n,dtype=np.int8))
            for name in ("sae-ce", "tabm"):
                r = subprocess.run(["torchrun","--standalone","--nproc_per_node=2","scripts/run_central_backbones.py","--model",name,
                                    "--out",str(root/name),"--cache",str(root),"--cpu","--batch","8","--micro","2","--epochs","1",
                                    "--seed","41"], capture_output=True, text=True, env=dict(os.environ, OMP_NUM_THREADS="1"))
                self.assertEqual(r.returncode, 0, r.stderr[-2000:])
                self.assertTrue((root/name/"epoch_001/weights.pt").exists())

    def test_models_and_strict_state(self):
        torch.set_num_threads(2)
        for name in ("mlp","tabm","bert","cnn-bigru-attention"):
            with self.subTest(model=name):
                m = build_model(name)
                x = torch.randn(3,77)
                y = torch.tensor([1,5,15])
                z = m(x)
                self.assertEqual(tuple(z.shape), (3,4,16) if name == "tabm" else (3,16))
                loss = classification_loss(z,y)
                if name == "tabm":
                    torch.testing.assert_close(loss, torch.stack([F.cross_entropy(v,y) for v in z.unbind(1)]).mean())
                    torch.testing.assert_close(predict(z), z.softmax(-1).mean(1).argmax(-1))
                loss.backward()
                self.assertTrue(all(p.grad is not None and p.grad.isfinite().all() for p in m.parameters()))
                clone = build_model(name).eval()
                clone.load_state_dict(m.state_dict(), strict=True)
                with torch.no_grad():
                    torch.testing.assert_close(clone(x), m.eval()(x))

    def test_attention_key_bias_nullspace(self):
        # q_i dot b_key is constant over key positions, so softmax cancels it.
        torch.manual_seed(17)
        attention = torch.nn.MultiheadAttention(40, 4, batch_first=True).double().eval()
        q, k, v = [torch.randn(2, 7, 40, dtype=torch.float64) for _ in range(3)]
        expected, _ = attention(q, k, v, need_weights=False)
        expected.square().sum().backward()
        self.assertLess(float(attention.in_proj_bias.grad[40:80].abs().max()), 1e-12)
        with torch.no_grad():
            attention.in_proj_bias[40:80].add_(torch.linspace(-3., 3., 40))
        actual, _ = attention(q, k, v, need_weights=False)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)

    def test_cnn_bigru_attention(self):
        model = build_model("cnn-bigru-attention").eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 118288)
        x = torch.randn(3, 77)
        features = model.conv(x.unsqueeze(1)).transpose(1, 2)
        self.assertEqual(tuple(features.shape), (3, 35, 128))
        states, _ = model.gru(features)
        scores = model.score(torch.tanh(model.attention(states))).squeeze(-1)
        weights = (scores - scores.max(1, keepdim=True).values).exp()
        weights = weights / weights.sum(1, keepdim=True)
        context = torch.bmm(weights.unsqueeze(1), states).squeeze(1)
        torch.testing.assert_close(model(x), model.head(context))
        # Sharded evaluation must reproduce the saved model, including BN buffers.
        torch.testing.assert_close(model(x), torch.cat([model(row[None]) for row in x]))

    def test_sae_objective_and_split_batch_gradient(self):
        from core.sae import batch_code_mean, sae_loss
        torch.manual_seed(42)
        model = build_model("sae")
        x = torch.randn(7, 77)
        x[:, 71:] = (x[:, 71:] > 0).float()
        y = torch.arange(7)
        logits, h, recon, target = model(x, auxiliary=True)
        self.assertEqual(tuple(logits.shape), (7, 16))
        torch.testing.assert_close(target[:, 71:], x[:, 71:])
        torch.testing.assert_close(logits, model.head(torch.cat((target, h), 1)))
        torch.testing.assert_close(model.eval()(x), logits)
        p = h.mean(0)
        kl = 0.05 * (0.05 / p).log() + 0.95 * (0.95 / (1-p)).log()
        expected = (recon-target).square().sum(1).mean() + 1.2 * kl.sum() + 1.5 * F.cross_entropy(logits,y)
        expected.backward()
        grads = [v.grad.clone() for v in model.parameters()]
        model.zero_grad()
        rows = torch.arange(len(x))
        mean = batch_code_mean(model, x, rows, 3, len(x), amp=False)
        for ix in rows.split(3):
            loss = sae_loss(model(x[ix], auxiliary=True), y[ix], len(ix), len(x), mean)
            (loss * len(ix) / len(x)).backward()
        for grad, param in zip(grads, model.parameters()):
            torch.testing.assert_close(grad, param.grad, atol=2e-6, rtol=2e-5)



if __name__ == "__main__":
    unittest.main()
