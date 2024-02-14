"""
Functions for plotting conformer ensembles and performing RMSD analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import pandas as pd
import seaborn as sns
from tqdm.notebook import tqdm
from functions import utils

import MDAnalysis as mda
from MDAnalysis.analysis import rms

# define default plotting colors
GRAY = '#D0D3C5'
BLUE = '#08708A'
RED = '#D73A31'
LIGHT_BLUE = '#56B1BF'


def bar_graph(ligands: str, dictionary: dict, save=False):
    """
    Plots a bar graph of the number of conformers for each ligand in a dictionary.
    
    Args:
        ligands (str): List of ligands.
        dictionary (dict): Dictionary of ligands and number of conformers.
        save (bool, optional): Whether to save the figure. Defaults to False.
        
    Returns:
        None
    """

    x = np.arange(len(ligands))
    width = 0.25
    multiplier = 0

    fig, ax = plt.subplots(layout="constrained", figsize=(13, 5))

    for attribute, measurement in dictionary.items():
        offset = multiplier * width
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_ylabel("Number of conformers")
    ax.set_xticks(x + width, ligands)
    ax.legend()

    if save:
        plt.savefig("conformer_bars.png")


def rmsd_analysis(ligands: str, set1: Path, set2: Path, save=False):
    """
    Performs an RMSD analysis on two sets of conformers.
    RMSD analysis is performed using MDAnalysis.
    See https://docs.mdanalysis.org/stable/documentation_pages/analysis/rms.html.
    Resultant RMSD analysis is printed to the console and plotted as a histogram.
    
    Args:
        ligands (str): List of ligands.
        set1 (Path): Path to first set of conformers.
        set2 (Path): Path to second set of conformers.
        save (bool, optional): Whether to save the figure. Defaults to False.

    Returns:
        None
    """

    filelist1 = os.listdir(set1)
    filelist2 = os.listdir(set2)

    fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(20, 20))

    file1 = None
    file2 = None

    for ligand, ax in zip(tqdm(ligands, total=len(ligands)), axs.ravel()[:len(ligands)]):
        for file in filelist1:
            if ligand in file:
                file1 = file

        for file in filelist2:
            if ligand in file:
                file2 = file

        u1 = mda.Universe(set1 / file1)
        u1.trajectory[0]

        rmsd_analysis1 = rms.RMSD(u1, select="all")
        rmsd_analysis1.run()

        u2 = mda.Universe(set2 / file2)
        u2.trajectory[0]

        rmsd_analysis2 = rms.RMSD(u2, select="all")
        rmsd_analysis2.run()

        df1 = pd.DataFrame(rmsd_analysis1.results.rmsd[:, 2:], columns=["all"], 
                           index=rmsd_analysis1.results.rmsd[:, 1])
        
        mean1 = df1["all"].mean()
        std1 = df1["all"].std()
        range1 = df1["all"].max() - df1["all"].min()
        print(f"{ligand} {set1.name} mean: {mean1}, std: {std1}, range: {range1}")

        df2 = pd.DataFrame(rmsd_analysis2.results.rmsd[:, 2:], columns=["all"], 
                           index=rmsd_analysis2.results.rmsd[:, 1])
        
        mean2 = df2["all"].mean()
        std2 = df2["all"].std()
        range2 = df2["all"].max() - df2["all"].min()
        print(f"{ligand} {set2.name} mean: {mean2}, std: {std2}, range: {range2}")

        ax.set_title(ligand, fontdict={'weight': 'bold'})
        
        data = {f"{set1.name}": df1["all"], f"{set2.name}": df2["all"]}
        sns.histplot(data=data, kde=True, stat="density", ax=ax)
        ax.set_xlabel("RMSD (Å)")
    
    plt.tight_layout()

    if save:
        plt.savefig(f"rmsd_analysis_{set1.name}_{set2.name}.png")


def feature_histograms(ligands: str, feature: str, data1: pd.DataFrame, data2: pd.DataFrame, label1: str, label2: str,
                       save=False):
    """
    Plots histograms of a feature for two sets of conformers.
    
    Args:
        ligands (str): List of ligands.
        feature (str): Feature to plot.
        data1 (pd.DataFrame): Dataframe containing feature data for set 1.
        data2 (pd.DataFrame): Dataframe containing feature data for set 2.
        label1 (str): Label for set 1.
        label2 (str): Label for set 2.
        save (bool, optional): Whether to save the figure. Defaults to False.
        
    Returns:
        None
    """

    fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(20, 20))

    for i, ax in zip(ligands, axs.ravel()[:len(ligands)]):
        data1_use = data1[data1['file'].str.contains(i)]
        data1_use.reset_index(drop=True, inplace=True)
        data2_use = data2[data2['file'].str.contains(i)]
        data2_use.reset_index(drop=True, inplace=True)
        
        ax.set_title(str(i), fontdict={'weight': 'bold'})
        
        data = {label1: data1_use[feature], label2: data2_use[feature].to_list()}

        sns.histplot(data, kde=True, stat="density", ax=ax)

        ax.set_xlabel(feature)

    plt.tight_layout()

    if save:
        plt.savefig(f"feature_histograms_{feature}.png")


def plot_equidistant_feature_selections(df, ligand_id, features, sele_num=10, text_file_path=None,
                                        write_text_file=False, savefig=False):
    """
    Plots equidistant selections of conformers based on a feature.
    Selections are plotted as a histogram and as a scatter plot of the feature against the xtb energy.
    The LEC is also plotted.
    Outputs a text file of the selected conformers.
    
    Args:
        df (pd.DataFrame): Dataframe containing feature data.
        ligand_id (str): Ligand ID.
        features (list): List of features to plot.
        sele_num (int, optional): Number of conformers to select. Defaults to 10.
        text_file_path (Path, optional): Path to text file. Defaults to None.
        write_text_file (bool, optional): Whether to write a text file. Defaults to False.
        savefig (bool, optional): Whether to save the figure. Defaults to False.
        
    Returns:
        None
    """

    global lec_selection
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(20, 15))
    fig.suptitle(f"Ligand {ligand_id}: Equidistant selections based on steric/geometric features", fontsize=16, x=0.5,
                 y=0.93, fontweight='bold')

    for i, ax in zip(features, axs.ravel()[3:6]):
        ax.plot(df[i].index, df[i], alpha=0.7, color=GRAY)
        ax.set_ylabel(i)
        selected = utils.select_equidistant_values(df, i, sele_num)
        lec_selection = utils.select_lec(df, 'xtb_energy')
        ax.scatter(selected.index, selected[i], color=BLUE, marker='.', s=300, edgecolor='k')
        ax.scatter(lec_selection.index, lec_selection[i], color=RED, marker='.', s=300, edgecolor='k')
        ax.set_xticks([])
        ax.set_xticklabels([])

        if write_text_file:
            with open(text_file_path / f"{ligand_id}_sterics_{sele_num}chosen.txt", 'a') as f:
                print(f"Selected ligands from {i}: ")
                f.write(f"Selected ligands from {i}: \n")
                for j in selected['ligand']:
                    print(j)
                for m in lec_selection['ligand']:
                    print(f"LEC: {m}")
                    f.write(f"LEC: {m}\n")
                print("-------------------------------------------\n")
                for j in selected['ligand']:
                    f.write(j + '\n')
        else:
            continue

    for i, ax in zip(features, axs.ravel()[:3]):
        ax.hist(df[i], alpha=0.5, label=f"All ({len(df['ligand'])})", color=GRAY)
        ax.set_xlabel(i)
        selected = utils.select_equidistant_values(df, i, sele_num)
        ax.hist(selected[i], color=BLUE, label=f"Selected ({sele_num})")
        ax.hist(lec_selection[i], color=RED, label='LEC')
        ax.legend()
        ax.set_ylabel('Count')

    for i, ax in zip(features, axs.ravel()[6:9]):
        df['rel_energy'] = (df['xtb_energy'] - df['xtb_energy'].min()) * 627.509
        df['f(E)'] = 1 / np.exp((df['rel_energy'] * 1000) / (1.987204 * 298.15))
        ax.scatter(df['rel_energy'], df['f(E)'], alpha=0.5, color=GRAY, label=f"All ({len(df['ligand'])})", s=100)
        selected = utils.select_equidistant_values(df, i, sele_num)
        lec_selection = utils.select_lec(df, 'xtb_energy')
        ax.scatter(selected['rel_energy'], selected['f(E)'], color=BLUE, marker='.', s=300, edgecolor='k',
                   label=f"Selected ({sele_num})")
        ax.scatter(lec_selection['rel_energy'], lec_selection['f(E)'], color=RED, marker='.', s=300, edgecolor='k',
                   label='LEC')
        ax.set_ylabel('$f(E)$')
        ax.set_xlabel('Relative energy / kcal mol$^{–1}$')
        ax.legend()

    if savefig:
        plt.savefig(f"{ligand_id}_steric-sel_{sele_num}confs.png")


def plot_equidistant_energy_selections(df: pd.DataFrame, ligand_id: str, features: list, sele_num: int = 10,
                                       text_file_path: Path = None, write_text_file: Path = False, savefig=False):
    """
    Plots equidistant selections of conformers based on the xtb energy.
    Selections are plotted as a histogram and as a scatter plot of the xtb energy against the xtb energy.
    The LEC is also plotted.
    Outputs a text file of the selected conformers.
    
    Args:
        df (pd.DataFrame): Dataframe containing feature data.
        ligand_id (str): Ligand ID.
        features (list): List of features to plot.
        sele_num (int, optional): Number of conformers to select. Defaults to 10.
        text_file_path (Path, optional): Path to text file. Defaults to None.
        write_text_file (bool, optional): Whether to write a text file. Defaults to False.
        savefig (bool, optional): Whether to save the figure. Defaults to False.
        
    Returns:
        None
    """

    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(20, 15))
    fig.suptitle(f"Ligand {ligand_id}: Equidistant selections based on GFN2-xTB energy", fontsize=16, x=0.5, y=0.93,
                 fontweight='bold')

    for i, ax in zip(features, axs.ravel()[:3]):
        df['rel_energy'] = (df['xtb_energy'] - df['xtb_energy'].min()) * 627.509
        df['f(E)'] = 1 / np.exp((df['rel_energy'] * 1000) / (1.987204 * 298.15))
        ax.scatter(df['rel_energy'], df['f(E)'], alpha=0.5, color=GRAY, s=100, label=f"All ({len(df['ligand'])})")
        selected = utils.select_equidistant_values(df, 'xtb_energy', sele_num)
        ax.scatter(selected['rel_energy'], selected['f(E)'], color=BLUE, marker='.', s=300, edgecolor='k',
                   label=f"Selected ({sele_num})")
        ax.set_ylabel('$f(E)$')
        ax.set_xlabel('Relative energy / kcal mol$^{–1}$')
        ax.legend()

    if write_text_file:
        with open(text_file_path / f"{ligand_id}_energy_{sele_num}chosen.txt", 'a') as f:
            print(f"Selected ligands from energy: ")
            f.write(f"Selected ligands from energy: \n")
            for j in selected['ligand']:
                print(j)
            print("-------------------------------------------\n")
            for j in selected['ligand']:
                f.write(j + '\n')

    for i, ax in zip(features, axs.ravel()[3:6]):
        ax.plot(df[i].index, df[i], alpha=0.7, color=GRAY)
        ax.set_ylabel(i)
        selected = utils.select_equidistant_values(df, 'xtb_energy', sele_num)
        ax.scatter(selected.index, selected[i], color=BLUE, marker='.', s=300, edgecolor='k')
        ax.set_xticks([])
        ax.set_xticklabels([])

    for i, ax in zip(features, axs.ravel()[6:9]):
        ax.hist(df[i], alpha=0.5, color=GRAY, label=f"All ({len(df[i])})")
        ax.set_xlabel(i)
        selected = utils.select_equidistant_values(df, 'xtb_energy', sele_num)
        ax.hist(selected[i], color=BLUE, label=f"Selected ({sele_num})")
        ax.legend()
        ax.set_ylabel('Count')

    if savefig:
        plt.savefig(f"{ligand_id}_energy-sel_{sele_num}confs.png")


def plot_dft_distributions(df_all, df_sele, ligand_id, descriptors, savefig=False):
    """
    Plots the distribution of DFT descriptors for a ligand.
    Selected conformers are plotted as a scatter plot.
    
    Args:
        df_all (pd.DataFrame): Dataframe containing all conformers.
        df_sele (pd.DataFrame): Dataframe containing selected conformers.
        ligand_id (str): Ligand ID.
        descriptors (list): List of descriptors to plot.
        savefig (bool, optional): Whether to save the figure. Defaults to False.
        
    Returns:
        None
    """

    fig, axs = plt.subplots(nrows=6, ncols=3, figsize=(20, 20))
    # fig.suptitle(f"Ligand {ligand_id} selected conformers", fontsize=16, fontweight='bold')

    for descriptors, ax in zip(descriptors, axs.ravel()):
        sns.histplot(df_all[descriptors], kde=True, ax=ax, color=LIGHT_BLUE)
        for i in df_sele[descriptors]:
            ax.scatter(x=i, y=0.1, color=RED, s=400, edgecolor='w')

    plt.tight_layout()

    if savefig:
        plt.savefig(f"{ligand_id}_dft_distribution.png")
