import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd

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


def plot_tsne(dim1: pd.DataFrame, dim2: pd.DataFrame, save=False):
    plt.figure(figsize=(8.5,8))
    plt.scatter(dim1, dim2)
    plt.xlabel(dim1.name)
    plt.ylabel(dim2.name)
    plt.xticks([])
    plt.yticks([])
    plt.show()

    if save:
        plt.savefig("tsne.svg")


def plot_tsne_clusters(dim1: pd.DataFrame, dim2: pd.DataFrame, clusters: pd.DataFrame, centroids=None, legend=True, save=False):
    plt.figure(figsize=(8.5,8))

    n_clusters = len(clusters.unique())

    for cluster in range(0, n_clusters):
        plt.scatter(dim1.loc[clusters == cluster+1],
                    dim2.loc[clusters == cluster+1],
                    label=f"Cluster {cluster+1}")
        
    if centroids is not None:
        for centroid in centroids:
            plt.scatter(dim1[centroid],
                        dim2[centroid],
                        marker='X',
                        s=200,
                        color='k')
        
    plt.xlabel(dim1.name)
    plt.ylabel(dim2.name)
    plt.xticks([])
    plt.yticks([])
    
    if legend:
        plt.legend()

    plt.show()

    if save:
        plt.savefig("tsne_clusters.svg")


def plot_pca(dim1: pd.DataFrame, dim2: pd.DataFrame, save=False):
    plt.figure(figsize=(8.5,8))
    plt.scatter(dim1, dim2)
    plt.xlabel(dim1.name)
    plt.ylabel(dim2.name)
    plt.show()

    if save:
        plt.savefig("pca.svg")


def plot_pca_clusters(dim1: pd.DataFrame, dim2: pd.DataFrame, clusters: pd.DataFrame, centroids=None, legend=False, save=False):
    plt.figure(figsize=(8.5,8))
    
    n_clusters = len(clusters.unique())

    for cluster in range(0, n_clusters):
        plt.scatter(dim1.loc[clusters == cluster+1],
                    dim2.loc[clusters == cluster+1],
                    label=f"Cluster {cluster+1}")
        
    if centroids is not None:
        for centroid in centroids:
            plt.scatter(dim1[centroid],
                        dim2[centroid],
                        marker='X',
                        s=200,
                        color='k')
        
    plt.xlabel(dim1.name)
    plt.ylabel(dim2.name)

    if legend:
        plt.legend()

    plt.show()

    if save:
        plt.savefig("pca_clusters.svg")