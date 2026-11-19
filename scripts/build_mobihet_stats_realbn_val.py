"""Matched train/val BN ablation, reusing the stats screening notebook template."""
import ast
import hashlib

import nbformat
import build_mobihet_stats_val as template

SLUG = "mobihet-v4-stats-realbn-val-r10"
CANDIDATE = "mobihet_stats_realbn"
DESCRIPTION = "Candidate keeps Gaussian CE unchanged; BN moments use real rows only."


def main():
    template.SLUG = SLUG
    template.main()
    path = template.ROOT / "notebooks" / template.SLUG / (template.SLUG + ".ipynb")
    nb = nbformat.read(path, 4)
    nb.cells[0].source = """# Stats real-only BatchNorm: paired train→val R10

Hypothesis: pooled synthetic rows distort local BatchNorm statistics. Candidate keeps identical Gaussian CE,
batch, optimizer and schedule, but BN moments use real rows only. Control is unchanged mobihet_stats.
Locked gate: macro +0.005; client mean/p10 >= -0.005; class13 >= -0.01; train overhead <=20%; round bytes zero.
Official test excluded. This is exploratory development after prior official-test inspection; that test is no longer
an untouched holdout for subsequent methods. Independent confirmation will be required.
"""
    for i, cell in enumerate(nb.cells):
        if i == 2 or cell.cell_type != "code":
            continue  # Embedded source and hashes stay byte-identical.
        cell.source = cell.source.replace('"mobihet_stats"', '"' + CANDIDATE + '"')
        cell.source = cell.source.replace('"mobihet"', '"mobihet_stats"')
        cell.source = cell.source.replace("mobihet-v4-control-stats-val-r10", "mobihet-v4-control-realbn-val-r10")
        cell.source = cell.source.replace("mobihet-v4-stats-val-r10", template.SLUG)
    nb.cells[0].source = f"""# {SLUG}

{DESCRIPTION}
Control: mobihet_stats. Paired train-only → full validation R10, 50-round cosine horizon.
Locked gate: macro +0.005; client mean/p10 >= -0.005; class13 >= -0.01; train overhead <=20%; round bytes zero.
Exploratory development after prior official-test inspection; independent confirmation required.
"""
    gate = nb.cells[7].source
    gate = gate.replace('deltas["macro_f1"] >= -0.005', 'deltas["macro_f1"] >= 0.005')
    gate = gate.replace('deltas["class13_f1"] >= 0', 'deltas["class13_f1"] >= -0.01')
    gate = gate.replace('time_reduction >= 0.30', 'time_reduction >= -0.20')
    gate = gate.replace('macro/mean/p10 >=-0.005; class13 >=0; train time -30%; per-round bytes=0',
                        'macro >=+0.005; mean/p10 >=-0.005; class13 >=-0.01; train overhead <=20%; round bytes=0')
    nb.cells[7].source = gate
    for cell in nb.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
    nbformat.validate(nb)
    nbformat.write(nb, path)
    print("FINAL SHA256", hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
