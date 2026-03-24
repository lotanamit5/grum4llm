import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

CRITERIA_LABELS = {
    "random": "Random",
    "d_opt": "D-optimality",
    "e_opt": "E-optimality",
    "social": "Social Choice",
    "personalized": "Personalized Choice"
}

STYLE_MAP_DEFAULT = {
    "random": {"color": "blue", "linestyle": "--"},
    "social": {"color": "red", "linestyle": "-"},
    "d_opt": {"color": "orange", "linestyle": "-."},
    "e_opt": {"color": "green", "linestyle": ":"},
    "personalized": {"color": "red", "linestyle": "-"},
}

def plot_elicitation(df, style_dict=None, window_size=1, ax=None):
    """
    Plots a line per criteria grouped by step, averaged over seeds without error deviations.
    Expects df with columns: step, criteria, value, seed.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    if style_dict is None:
        style_dict = STYLE_MAP_DEFAULT
        
    avg_df = df.groupby(["step", "criteria"])["value"].mean().reset_index()
    
    for crit in avg_df["criteria"].unique():
        crit_df = avg_df[avg_df["criteria"] == crit].sort_values("step")
        style = style_dict.get(crit, {"color": "black", "linestyle": "-"})
        label = CRITERIA_LABELS.get(crit, crit)
        
        if window_size > 1:
            y_vals = crit_df["value"].rolling(window=window_size, min_periods=1).mean().values
        else:
            y_vals = crit_df["value"].values
            
        ax.plot(crit_df["step"], y_vals, label=label, **style)
        
    ax.set_xlabel("Number of Agents")
    ax.set_ylabel("Kendall-Correlation")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    return ax

def plot_asymptotic(df, metrics_columns: list[str], style_dict=None, ax=None):
    """
    Plots a series of box plots on the x-axis per step for all metric columns.
    Expects df with columns: step, seed, and raw metric columns mapped explicitly.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        
    melted = df.melt(id_vars=["step", "seed"], value_vars=metrics_columns, var_name="Metric", value_name="value")
    
    palette = None
    if style_dict is not None:
        palette = {m: style_dict.get(m, {}).get("color", "gray") for m in metrics_columns}
        
    sns.boxplot(data=melted, x="step", y="value", hue="Metric", ax=ax, palette=palette)
    
    ax.set_xlabel("Number of Agents")
    ax.set_ylabel("Kendall-Correlation")
    ax.grid(True, alpha=0.3)
    
    # Improve tick spacing dynamically tracking the massive steps efficiently
    step_unique = sorted(melted["step"].unique())
    if len(step_unique) > 15:
        ticks = ax.get_xticks()
        labels = ax.get_xticklabels()
        new_ticks, new_labels = [], []
        for i, (t, l) in enumerate(zip(ticks, labels)):
            if i % (len(step_unique) // 10) == 0:
                new_ticks.append(t)
                new_labels.append(l)
        ax.set_xticks(new_ticks)
        ax.set_xticklabels(new_labels)

    return ax
