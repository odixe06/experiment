"""MobiHet ablation on one mixed-model case: scripts/run_mixed_l4.py unchanged, with the ablation method
(methods/mobihet_ablation_mixed.py) and the arm's hp on top of MOBIHET_HP. Same CLI plus --ablation.

    torchrun --standalone --nproc_per_node=2 scripts/run_mixed_ablation.py --ablation a1 --case case2 --seed 42 --out runs/x --compile
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from methods.mobihet_ablation import HP as ABLATION_HP
from methods.mobihet_ablation_mixed import Method
from scripts import run_mixed_l4 as R


def main():
    argv = sys.argv[1:]
    assert argv.count("--ablation") == 1, "--ablation <none|a1|a2|a4> is required"
    i = argv.index("--ablation")
    arm = argv[i + 1]
    assert arm in ABLATION_HP, arm
    del argv[i:i + 2]
    l4_hp = R.l4_hp
    R.l4_hp = lambda seed, overrides=(): l4_hp(seed, [*ABLATION_HP[arm], *overrides])
    R.Method = Method
    sys.argv = [str(ROOT / "scripts/run_mixed_l4.py"), *argv]
    R.main()


if __name__ == "__main__":
    main()
