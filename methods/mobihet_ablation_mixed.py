"""MobiHet L4 ablations on a mixed-model case: methods.mobihet_mixed_l4 with every backbone group built as
Ablation + GroupL4 (methods/mobihet_ablation.py) instead of GroupL4. Lockstep, assignment, evaluation and state unchanged."""
from methods import mobihet_mixed_l4 as M
from methods.mobihet_ablation import Ablation


class GroupAblation(Ablation, M.GroupL4):
    pass


class Method(M.Method):
    def __init__(self, env, assignment, models):
        group = M.GroupL4                                                # M.Method builds its groups by this module name
        M.GroupL4 = GroupAblation
        try:
            super().__init__(env, assignment, models)
        finally:
            M.GroupL4 = group
