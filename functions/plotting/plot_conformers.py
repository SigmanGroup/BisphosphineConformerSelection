import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import MDAnalysis as mda
from MDAnalysis.analysis import rms

def bar_graph(ligands: str, dictionary: dict, save=False):
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
        plt.savefig("conformers.png")


def rmsd_analysis(ligands: str, set1: Path, set2: Path, save=False):
    filelist1 = os.listdir(set1)
    filelist2 = os.listdir(set2)

    fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(20, 20))

    for ligand, ax in zip(tqdm(ligands, ncols=80), axs.ravel()[:len(ligands)]):
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
        plt.savefig(f"rmsd_analysis_{set1.name}_{set2.name}.svg")


def feature_histograms(ligands: str, feature: str, data1: pd.DataFrame, data2: pd.DataFrame, label1: str, label2: str, save=False):
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
        plt.savefig(f"feature_histograms_{feature}.svg")