import matplotlib.pyplot as plt
from grums.utils.plot_utils import smooth_curve

CRITERIA_LABELS = {
    "random": "Random",
    "d_opt": "D-optimality",
    "e_opt": "E-optimality",
    "social": "Social Choice",
    "personalized": "Personalized Choice"
}

def plot_asymptotic_figure(df, metric: str, title: str, ax=None):
    """Renders robust, non-smoothed performance boundaries tracking massive iterations."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        
    df_metric = df[df["metric"] == metric]
    for criterion in df_metric["criterion"].unique():
        df_crit = df_metric[df_metric["criterion"] == criterion].sort_values("n_observations")
        label = CRITERIA_LABELS.get(criterion, criterion)
        
        ax.plot(df_crit["n_observations"], df_crit["mean"], label=label, linewidth=2)
        ax.fill_between(
            df_crit["n_observations"], 
            df_crit["mean"] - df_crit["std"], 
            df_crit["mean"] + df_crit["std"], 
            alpha=0.2
        )
        
    ax.set_xlabel("Number of Ranking Observations")
    ax.set_ylabel("Kendall Tau Evaluation Value")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax

def plot_elicitation_figure(df, metric: str, criterias: list[str], window_size: int, title: str, ax=None):
    """Renders explicitly smoothed interval curves isolating specified criteria evaluations smoothly mapping variances."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        
    df_metric = df[df["metric"] == metric]
    for criterion in criterias:
        df_crit = df_metric[df_metric["criterion"] == criterion]
        if df_crit.empty:
            continue
            
        df_crit = df_crit.sort_values("n_observations")
        y_mean = smooth_curve(df_crit["mean"].values, window_size)
        y_std = smooth_curve(df_crit["std"].values, window_size)
            
        label = CRITERIA_LABELS.get(criterion, criterion)
        ax.plot(df_crit["n_observations"], y_mean, label=label, linewidth=2)
        ax.fill_between(
            df_crit["n_observations"], 
            y_mean - y_std, 
            y_mean + y_std, 
            alpha=0.2
        )
        
    ax.set_xlabel("Number of Ranking Observations")
    ax.set_ylabel("Kendall Tau Evaluation Value")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    return ax
