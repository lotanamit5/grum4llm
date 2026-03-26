Part 1: Context & Setup
Introduction – What this notebook is about; how it differs from 

1_reproduction.ipynb
 (which reproduced the original GRUM paper). Here we apply GRUM to LLM preference elicitation using the Colors domain as a validation testbed.
Data Loading – Load all 12 result JSON files from llm_colors-20260326-114531, parse metadata (model, embedding, criterion), extract final-step GRUM delta + B matrix, BT delta.
Part 2: Intrinsic Preferences — Experiment 1
GRUM δ vs BT β Correlation – Bar chart + scatter: do GRUM's intrinsic scores align with BT? Across all 4 model×embedding combos. Conclude on conjecture.
Preference Rankings Table – Rank-order of colors for each config (GRUM vs BT side-by-side).
Part 3: Embedding Method Comparison
Embedding Type Effect on δ – Side-by-side bars: HS vs ST embeddings, per model type. Do they produce similar intrinsic preferences?
Pretrained vs Instruct Effect – Compare resulting δ and interaction strength.
Part 4: Elicitation Criterion Comparison
Criterion Effect on δ – Social vs Personalized vs Random: do they converge to the same preferences?
Training Convergence by Criterion – Plot delta/NLL over training steps for each criterion.
Part 5: Persona Effect — Experiment 2
Interaction Matrix B Heatmaps – Visualize B for all configs; does it contain structure?
Column Similarity Analysis – Quantify if B ≈ global bias (all cols similar) or genuine item-specific persona interaction.
Rank Reversals Under Persona – Simulate "frugal" vs "luxury" hidden states; show preference reversals between items.
Part 6: Predictive Performance — Experiment 3
NLL Comparison: GRUM vs BT – Compare held-out NLL; does GRUM beat BT?
NLL Convergence Curves – Show training dynamics.
Part 7: Summary
Summary Table – All key metrics in one table (correlation, avg col-sim, NLL gap) across all configurations.
Conclusions – Narrative summary of all three conjectures.
