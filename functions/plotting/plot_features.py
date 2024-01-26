import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def feature_bars(feature_list, feature_values, ax, title=None, xlabel=None):
    ax.barh(feature_list, feature_values)
    ax.invert_yaxis()
    ax.set_title(title, fontdict={'weight': 'bold'})
    ax.set_xlabel(xlabel)

def plot_regression(x, y, ax):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    sns.regplot(x=x, y=y, ax=ax)
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    plt.text(0.05, 0.95, f"$R^2$ = {round(r_value**2, 2)}\n$p$-value = {p_value:.2e}", transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=props)