import matplotlib.pyplot as plt
import seaborn as sns
from functions import utils
import numpy as np
from matplotlib.font_manager import fontManager, FontProperties
from scipy import stats

# path = '/usr/local/share/fonts/AvenirLTStd-Black.otf'
# fontManager.addfont(path)
# prop = FontProperties(fname=path)
# sns.set(font=prop.get_name())
# sns.set_theme(context='notebook', style='ticks', font=prop.get_name(), font_scale=1.3)

GRAY = '#D0D3C5'
BLUE = '#08708A'
RED = '#D73A31'
HIST_COLOR = '#56B1BF'


def plot_equidistant_feature_selections(df, ligand_id, features, sele_num=10, text_file_path=None, write_text_file=False, savefig=False):
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(20,15))
    fig.suptitle(f"Ligand {ligand_id}: Equidistant selections based on steric/geometric features", fontsize=16, x=0.5, y=0.93, fontweight='bold')

    for i, ax in zip(features, axs.ravel()[3:6]):
        ax.plot(df[i].index, df[i], alpha=0.7, color=GRAY)
        ax.set_ylabel(i)
        selected = utils.select_equidistant_values(df, i, sele_num)
        lec_selection = utils.select_lec(df, 'xtb_energy')
        ax.scatter(selected.index, selected[i], color=BLUE, marker='.', s=300, edgecolor='k')
        ax.scatter(lec_selection.index, lec_selection[i], color=RED, marker='.', s=300, edgecolor='k')
        ax.set_xticks([])
        ax.set_xticklabels([])

        if write_text_file == True:
            with open(text_file_path / f"{ligand_to_analyze}_sterics_{conformers_to_select}chosen.txt", 'a') as f:
                print(f"Selected ligands from {i}: ")
                f.write(f"Selected ligands from {i}: \n")
                for j in selected['ligand']: print(j)
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
        ax.scatter(selected['rel_energy'], selected['f(E)'], color=BLUE, marker='.', s=300, edgecolor='k', label=f"Selected ({sele_num})")
        ax.scatter(lec_selection['rel_energy'], lec_selection['f(E)'], color=RED, marker='.', s=300, edgecolor='k', label='LEC')
        ax.set_ylabel('$f(E)$')
        ax.set_xlabel('Relative energy / kcal mol$^{–1}$')
        ax.legend()

    if savefig == True:
        plt.savefig(f"{ligand_id}_steric-sel_{sele_num}confs.svg")


def plot_equidistant_energy_selections(df, ligand_id, features, sele_num=10, write_text_file=False, savefig=False):
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(20,15))
    fig.suptitle(f"Ligand {ligand_id}: Equidistant selections based on GFN2-xTB energy", fontsize=16, x=0.5, y=0.93)

    for i, ax in zip(features, axs.ravel()[:3]):
        df['rel_energy'] = (df['xtb_energy'] - df['xtb_energy'].min()) * 627.509
        df['f(E)'] = 1 / np.exp((df['rel_energy'] * 1000) / (1.987204 * 298.15))
        ax.scatter(df['rel_energy'], df['f(E)'], alpha=0.5, color=GRAY, s=100, label=f"All ({len(df['ligand'])})")
        selected = utils.select_equidistant_values(df, 'xtb_energy', sele_num)
        ax.scatter(selected['rel_energy'], selected['f(E)'], color=BLUE, marker='.', s=300, edgecolor='k', label=f"Selected ({sele_num})")
        ax.set_ylabel('$f(E)$')
        ax.set_xlabel('Relative energy / kcal mol$^{–1}$')
        ax.legend()

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

    if savefig == True:
        plt.savefig(f"{ligand_id}_energy-sel_{sele_num}confs.svg")


def plot_dft_distributions(df_all, df_sele, ligand_id, descriptors, savefig=False):
    fig, axs = plt.subplots(nrows=6, ncols=3, figsize=(20,32))
    fig.suptitle(f"Ligand {ligand_id} selected conformers", fontsize=16, x=0.5, y=0.90)

    for descriptors, ax in zip(descriptors, axs.ravel()):
        sns.histplot(df_all[descriptors], kde=True, ax=ax, color=HIST_COLOR)
        for i in df_sele[descriptors]:
            ax.scatter(x=i, y=0.1, color=RED, s=400, edgecolor='w')

    if savefig == True:
        plt.savefig(f"{ligand_id}_dft_distribution.svg")


def plot_regression(x, y, ax):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    sns.regplot(x, y, ax=ax, color=BLUE)
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    plt.text(0.05, 0.95, f"$R^2$ = {round(r_value**2, 2)}\n$p$-value = {p_value:.2e}", transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=props)