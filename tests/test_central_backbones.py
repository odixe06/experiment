import unittest
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import numpy as np
import torch
from torch.nn import functional as F
from core.central_backbones import build_model, classification_loss, predict
from core.central_windows import predecessors, batch_input


class CentralModelsTest(unittest.TestCase):
    def test_parquet_to_windows_alignment(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from core import data
        from core.central_windows import build_index
        with tempfile.TemporaryDirectory(dir=".") as work:
            root = Path(work)
            cache = root/"cache"
            (root/"train").mkdir()
            for file, times in enumerate(([3,1,2],[1,4,5])):
                # Receiver 1 consumes rows at positions 0 and 2 in each parquet.
                values = [file*3+i+1 for i in range(3)]
                cols = {name: np.array(values,dtype="float32")/2 for name in data.ORDER}
                cols.update(client_id=[1,0,1],sender_id=[7]*3,rcv_time_ns=times,
                            message_id=values,fold=["train"]*3,label=values)
                pq.write_table(pa.table(cols), root/"train"/f"part-{file}.parquet")
            data.build_npy(str(root), str(cache), splits=("train",))
            build_index(root,"train",cache)
            y = torch.from_numpy(np.load(cache/"train_Y.npy")).reshape(-1,1)
            prev = torch.from_numpy(np.load(cache/"train_prev.npy"))
            result = batch_input(y,torch.arange(len(y)),prev)[:,-4:,0].tolist()
            self.assertEqual(result, [[0,0,0,2],[0,0,2,5],[0,4,3,1],
                                       [0,0,4,3],[0,0,0,4],[4,3,1,6]])

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
                prev = np.arange(-1,n-1,dtype=np.int32)
                np.save(root/f"{split}_prev.npy", np.stack([prev,prev],axis=1))
            for name in ("mlp","tabm","bert","cnn-lstm","cnn-lstm-feature","tcnn","grande","sae","gru-attention","cnn-bigru-attention","mambacnn"):
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
                if name in ("tabm","grande","sae","gru-attention","cnn-bigru-attention","mambacnn","cnn-lstm-feature"):
                    continuous = root/f"continuous-{name}"
                    launch([str(continuous) if v == str(out) else v for v in args])
                    expected = torch.load(continuous/"epoch_002/weights.pt",weights_only=True)
                    for key,tensor in torch.load(final/"weights.pt",weights_only=True).items():
                        self.assertTrue(torch.equal(expected[key],tensor))

    def test_gru_attention_pca(self):
        model = build_model("gru-attention")
        x = np.random.default_rng(5).normal(size=(160, 77))
        model.fit_pca(x)
        _, _, vh = np.linalg.svd(x - x.mean(axis=0), full_matrices=False)
        basis = model.components.numpy()
        np.testing.assert_allclose(model.center.numpy(), x.mean(axis=0), atol=1e-7)
        np.testing.assert_allclose(basis @ basis.T, vh[:5].T @ vh[:5], atol=1e-6)
        logits = model(torch.tensor(x[:3], dtype=torch.float32))
        torch.nn.functional.cross_entropy(logits, torch.tensor([1, 5, 15])).backward()
        self.assertEqual(tuple(logits.shape), (3, 16))
        for p in model.parameters():
            self.assertTrue(torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0)

    def test_models_and_strict_state(self):
        torch.set_num_threads(2)
        for name in ("mlp","tabm","bert","cnn-lstm","cnn-lstm-feature","tcnn","grande","cnn-bigru-attention","mambacnn"):
            with self.subTest(model=name):
                m = build_model(name)
                x = torch.randn((3,10,77) if name == "cnn-lstm" else (3,77))
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

    def test_cnn_lstm_feature_input_contract(self):
        torch.set_num_threads(2)
        model = build_model("cnn-lstm-feature").eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 196240)
        x = torch.randn(3, 77)
        with torch.no_grad():
            expected = model(x)
            # Each message is independent of other rows and has no temporal-history input.
            torch.testing.assert_close(expected, torch.cat([model(row[None]) for row in x]),
                                       atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(model(x.flip(0)), expected.flip(0))
        self.assertEqual(tuple(expected.shape), (3, 16))
        with self.assertRaises(ValueError):
            model(torch.randn(3, 10, 77))
        with self.assertRaises(RuntimeError):
            model.load_state_dict(build_model("cnn-lstm").state_dict(), strict=True)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA fp16")
    def test_cnn_lstm_feature_cuda_amp(self):
        model = build_model("cnn-lstm-feature").cuda().train()
        x = torch.randn(5, 77, device="cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(x)
        classification_loss(logits, torch.tensor([0, 1, 7, 13, 15], device="cuda")).backward()
        self.assertTrue(torch.isfinite(logits).all())
        for param in model.parameters():
            self.assertTrue(param.grad is not None and torch.isfinite(param.grad).all())

    def test_mamba_selective_scan(self):
        from core.mambacnn import selective_scan
        torch.manual_seed(17)
        # Non-power-of-two length and saturated decays catch scan padding/underflow errors.
        for dtype, tol in ((torch.float64, 1e-10), (torch.float32, 2e-5)):
            for length in (1, 7, 77):
                u = torch.randn(2, length, 3, dtype=dtype, requires_grad=True)
                dt = torch.rand_like(u).requires_grad_()
                a = -torch.arange(1, 5, dtype=dtype).expand(3, -1).clone().requires_grad_()
                b = torch.randn(2, length, 4, dtype=dtype, requires_grad=True)
                c = torch.randn_like(b, requires_grad=True)
                skip = torch.randn(3, dtype=dtype, requires_grad=True)
                args = (u, dt, a, b, c, skip)
                state = torch.zeros(2, 3, 4, dtype=dtype)
                values = []
                for t in range(length):
                    decay = torch.exp(torch.einsum("bh,hn->bhn", dt[:, t], a))
                    drive = torch.einsum("bh,bn,bh->bhn", dt[:, t], b[:, t], u[:, t])
                    state = decay * state + drive
                    values.append(torch.einsum("bhn,bn->bh", state, c[:, t]) + skip*u[:, t])
                expected = torch.stack(values, 1)
                actual = selective_scan(*args)
                torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)
                probe = torch.randn_like(actual)
                ref_grad = torch.autograd.grad((expected*probe).sum(), args)
                # With length 1 and zero initial state, a has no influence.
                got_grad = torch.autograd.grad((actual*probe).sum(), args, materialize_grads=True)
                for got, ref in zip(got_grad, ref_grad):
                    torch.testing.assert_close(got, ref, atol=tol, rtol=tol)

    def test_mambacnn_structure_and_amp(self):
        from core.mambacnn import MambaBlock
        torch.set_num_threads(2)
        torch.manual_seed(42)
        model = build_model("mambacnn").eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 129328)
        x = torch.linspace(-3.72, 3.72, 3*77).reshape(3,77)
        with torch.no_grad():
            torch.testing.assert_close(model(x), torch.cat([model(row[None]) for row in x]),
                                       atol=2e-6, rtol=2e-5)
        # The SSM convolution/scan is causal along feature positions.
        block = MambaBlock().eval()
        tokens = torch.randn(2, 13, 40)
        changed = tokens.clone()
        changed[:, 7:] += 4
        torch.testing.assert_close(block(tokens)[:, :7], block(changed)[:, :7])
        # CUDA workers exercise actual fp16; CPU uses bf16 to cover the AMP boundary.
        cuda_scan = torch.cuda.is_available() and importlib.util.find_spec("mamba_ssm") is not None
        dev = torch.device("cuda:0" if cuda_scan else "cpu")
        model = model.to(dev).train()
        with torch.autocast(dev.type, dtype=torch.float16 if dev.type == "cuda" else torch.bfloat16):
            logits = model(x.to(dev))
        classification_loss(logits, torch.tensor([0,7,15],device=dev)).backward()
        for param in model.parameters():
            self.assertTrue(param.grad is not None and param.grad.isfinite().all())

    @unittest.skipUnless(torch.cuda.is_available() and importlib.util.find_spec("mamba_ssm") is not None,
                         "requires official CUDA scan")
    def test_mamba_cuda_scan_equivalence(self):
        from core.mambacnn import selective_scan, selective_scan_reference
        torch.manual_seed(17)
        for length in (1, 7, 77):
            args = [torch.randn(3, length, 80, device="cuda"),
                    torch.rand(3, length, 80, device="cuda"),
                    -torch.arange(1, 17, device="cuda").float().repeat(80, 1),
                    torch.randn(3, length, 16, device="cuda"),
                    torch.randn(3, length, 16, device="cuda"),
                    torch.randn(80, device="cuda")]
            args = [v.requires_grad_() for v in args]
            expected = selective_scan_reference(*args)
            actual = selective_scan(*args)
            torch.testing.assert_close(actual, expected, atol=3e-5, rtol=1e-4)
            probe = torch.randn_like(actual)
            ref = torch.autograd.grad((expected*probe).sum(), args, materialize_grads=True)
            got = torch.autograd.grad((actual*probe).sum(), args)
            for left, right in zip(got, ref):
                torch.testing.assert_close(left, right, atol=3e-4, rtol=3e-4)

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

    def test_grande_routing_and_gradient(self):
        from core.grande import Entmax15, GRANDE
        torch.manual_seed(17)
        scores = torch.randn(3, 7, dtype=torch.double, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(Entmax15.apply, (scores,)))
        probs = Entmax15.apply(scores)
        torch.testing.assert_close(probs.sum(-1), torch.ones_like(probs[:,0]))
        model = GRANDE()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 82624)
        x = torch.randn(4, 77)
        with torch.no_grad():
            # Include an exact threshold tie in the root of tree 0.
            col = model.feature_scores[0,0].argmax()
            x[0,col] = model.thresholds[0,0,col]
            expected = []
            for row in x:
                weights, values = [], []
                for t in range(32):
                    node = 0
                    for _ in range(4):
                        col = model.feature_scores[t,node].argmax()
                        node = 2*node + (1 if row[col] < model.thresholds[t,node,col] else 2)
                    weights.append(model.leaf_weights[t,node-15])
                    values.append(model.leaf_logits[t,node-15])
                expected.append(torch.stack(weights).softmax(0) @ torch.stack(values))
            torch.testing.assert_close(model(x), torch.stack(expected))
        classification_loss(model(x), torch.arange(4)).backward()
        for p in model.parameters():
            self.assertTrue(p.grad.isfinite().all() and p.grad.abs().sum() > 0)

    def test_causal_windows(self):
        # Unsorted rows, two starts, two senders, two folds and two simultaneous past messages.
        client = np.array([0,0,0,0,1,0,0])
        sender = np.array([1,1,1,2,1,1,1])
        times = np.array([3,1,1,2,2,4,6])
        folds = np.array([0,0,0,0,0,0,1])
        idx = predecessors(client,sender,times,np.arange(7),folds)
        x = torch.arange(1,8)[:,None]
        windows = batch_input(x, torch.arange(7), torch.from_numpy(idx)).squeeze(-1)
        self.assertEqual(windows[5,-4:].tolist(), [2,3,1,6])
        for i in (1,2,3,4,6):
            self.assertEqual(windows[i].tolist(), [0]*9+[i+1])


if __name__ == "__main__":
    unittest.main()
