import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from pathlib import Path

nb = new_notebook()

nb.cells.extend([
    new_markdown_cell("# Reproducing Original GRUMs Figures\nThis notebook recreates Figures 2, 3, 4, 5, and 6 natively analyzing the active evaluation datasets evaluated through `asymptotic_repro` and `elicitation_repro` Slurm configurations."),
    new_code_cell("""\
import sys
from pathlib import Path
ROOT = Path('.').resolve().parents[0]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

import matplotlib.pyplot as plt
from grums.utils.plot_utils import load_metrics_for_datasets
from grums.utils.plots import plot_asymptotic_figure, plot_elicitation_figure

# Define canonical roots corresponding to orchestration execution folders
ASYM_DIR = ROOT / "results/repro/asymptotic_repro"
ELIC_DIR = ROOT / "results/repro/elicitation_repro"
"""),

    new_markdown_cell("## Figure 2: Social Choice Correlation (Asymptotic)"),
    new_code_cell("""\
# Dataset0: social-corr metric over 500 steps. No smoothing.
df_asym = load_metrics_for_datasets(ASYM_DIR, datasets=["ds0"])
if not df_asym.empty:
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_asymptotic_figure(df_asym, metric="social_tau", title="Figure 2: Social Choice Consistency", ax=ax)
    plt.show()
else:
    print("Data not found for Figure 2. Run the asymptotic_orchestration.yml pipeline first.")
"""),

    new_markdown_cell("## Figure 6: Personalized Choice Correlation (Asymptotic)"),
    new_code_cell("""\
# Dataset0: tracks 'mean_person_tau' globally evaluating exact user parameter distributions natively and 'raw_person_tau'.
if not df_asym.empty:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    plot_asymptotic_figure(df_asym, metric="mean_person_tau", title="Figure 6a: Personalized Pred. Correlation", ax=ax1)
    plot_asymptotic_figure(df_asym, metric="raw_person_tau", title="Figure 6b: Personalized Raw Correlation", ax=ax2)
    plt.show()
"""),

    new_markdown_cell("## Figure 3: Social Choice Correlation (Elicitation)"),
    new_code_cell("""\
# Datasets ds1, ds2. Evaluates social-corr utilizing sliding smoothers explicitly focusing on E-opt, D-opt, Random, and Social criteria bounds natively
df_elic_ds = load_metrics_for_datasets(ELIC_DIR, datasets=["ds1", "ds2"])
criterias = ["e_opt", "d_opt", "random", "social"]

if not df_elic_ds.empty:
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_elicitation_figure(df_elic_ds, metric="social_tau", criterias=criterias, window_size=10, title="Figure 3: Social Choice Elicitation", ax=ax)
    plt.show()
else:
    print("Data not found for Figure 3. Run the elicitation_orchestration.yml pipeline first.")
"""),

    new_markdown_cell("## Figure 4: Personalized Choice Correlation (Elicitation)"),
    new_code_cell("""\
# Datasets ds1, ds2. Identical configuration plotting mean personalized tau smoothly tracking exactly 90 active learning intervals smoothly.
if not df_elic_ds.empty:
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_elicitation_figure(df_elic_ds, metric="mean_person_tau", criterias=criterias, window_size=10, title="Figure 4: Personalized Choice Elicitation", ax=ax)
    plt.show()
"""),

    new_markdown_cell("## Figure 5: Sushi Application"),
    new_code_cell("""\
# Data: sushi exclusively. Analyzes social and mean personalized criteria natively alongside identical interval evaluation benchmarks.
df_sushi = load_metrics_for_datasets(ELIC_DIR, datasets=["sushi"])
if not df_sushi.empty:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    plot_elicitation_figure(df_sushi, metric="social_tau", criterias=criterias, window_size=10, title="Figure 5a: Sushi Social Choice", ax=ax1)
    plot_elicitation_figure(df_sushi, metric="mean_person_tau", criterias=criterias, window_size=10, title="Figure 5b: Sushi Personalized Choice", ax=ax2)
    plt.show()
else:
    print("Sushi execution payload missing. Execute elicitation pipeline natively evaluating `sushi` to populate graph.")
""")
])

output_file = Path("notebooks/repro.ipynb")
output_file.parent.mkdir(parents=True, exist_ok=True)
with open(output_file, 'w') as f:
    nbformat.write(nb, f)
print(f"Notebook successfully written to {output_file}")
