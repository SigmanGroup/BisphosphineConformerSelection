"""
Functions for plotting features in TSNE and PCA space.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd


def feature_bars(feature_list, feature_values, ax, title=None, xlabel=None):
    """
    Plots a horizontal bar chart of feature values.
    
    Args:
        feature_list (list): List of feature names.
        feature_values (list): List of feature values.
        ax (matplotlib.axes.Axes): Axes to plot on.
        title (str, optional): Title of plot. Defaults to None.
        xlabel (str, optional): Label for x-axis. Defaults to None.
        
    Returns:
        None
    """

    ax.barh(feature_list, feature_values)
    ax.invert_yaxis()
    ax.set_title(title, fontdict={'weight': 'bold'})
    ax.set_xlabel(xlabel)


def plot_regression(x, y, ax):
    """
    Plots a regression line on a scatter plot between two sets of features (CREST vs DFT-refined).
    r^2 and p-value are also displayed on the plot.
    
    Args:
        x (list): List of x values.
        y (list): List of y values.
        ax (matplotlib.axes.Axes): Axes to plot on.
        
    Returns:
        None
    """

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    sns.regplot(x=x, y=y, ax=ax)
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    plt.text(0.05, 0.95, f"$R^2$ = {round(r_value**2, 2)}\n$p$-value = {p_value:.2e}", transform=ax.transAxes,
             fontsize=12, verticalalignment='top', bbox=props)


def plot_tsne(dim1: pd.DataFrame, dim2: pd.DataFrame, save=False):
    """
    Plots a scatter plot TSNE feature space.
    
    Args:
        dim1 (pd.DataFrame): First dimension of TSNE feature space.
        dim2 (pd.DataFrame): Second dimension of TSNE feature space.
        save (bool, optional): Whether to save the plot. Defaults to False.
        
    Returns:
        None
    """

    plt.figure(figsize=(8.5, 8))
    plt.scatter(dim1, dim2)
    plt.xlabel(dim1.name)
    plt.ylabel(dim2.name)
    plt.xticks([])
    plt.yticks([])
    plt.show()

    if save:
        plt.savefig("tsne.png")


def plot_tsne_clusters(dim1: pd.DataFrame, dim2: pd.DataFrame, clusters: pd.DataFrame, centroids=None, legend=True,
                       save=False):
    """
    Plots a scatter plot of TSNE feature space with clusters.
    
    Args:
        dim1 (pd.DataFrame): First dimension of TSNE feature space.
        dim2 (pd.DataFrame): Second dimension of TSNE feature space.
        clusters (pd.DataFrame): Cluster labels.
        centroids (list, optional): List of cluster centroids. Defaults to None.
        legend (bool, optional): Whether to display the legend. Defaults to True.
        save (bool, optional): Whether to save the plot. Defaults to False.
        
    Returns:
        None
    """

    plt.figure(figsize=(8.5, 8))

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
        plt.savefig("tsne_clusters.png")


def plot_pca(dim1: pd.DataFrame, dim2: pd.DataFrame, save=False):
    """
    Plots a scatter plot of PCA feature space.
    
    Args:
        dim1 (pd.DataFrame): First dimension of PCA feature space.
        dim2 (pd.DataFrame): Second dimension of PCA feature space.
        save (bool, optional): Whether to save the plot. Defaults to False.
        
    Returns:
        None
    """

    plt.figure(figsize=(8.5, 8))
    plt.scatter(dim1, dim2)
    plt.xlabel(dim1.name)
    plt.ylabel(dim2.name)
    plt.show()

    if save:
        plt.savefig("pca.png")


def plot_pca_clusters(dim1: pd.DataFrame, dim2: pd.DataFrame, clusters: pd.DataFrame, centroids=None, legend=False,
                      save=False):
    """
    Plots a scatter plot of PCA feature space with clusters.
    
    Args:
        dim1 (pd.DataFrame): First dimension of PCA feature space.
        dim2 (pd.DataFrame): Second dimension of PCA feature space.
        clusters (pd.DataFrame): Cluster labels.
        centroids (list, optional): List of cluster centroids. Defaults to None.
        legend (bool, optional): Whether to display the legend. Defaults to False.
        save (bool, optional): Whether to save the plot. Defaults to False.
        
    Returns:
        None
    """

    plt.figure(figsize=(8.5, 8))
    
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
        plt.savefig("pca_clusters.png")
        