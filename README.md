# BisphosphineConformerSelection

[add introductory remarks here]

## Running the scripts

The scripts used in this work have been tested in Python version 3.11.7. In order to run the scripts, set up a Python environment following the below instructions:

### Using conda environment.yml file

- Run `conda env create -n conf_selection --file environment.yml`

### Using requirements.txt file

- Set up a conda environment (Python 3.11.7): `conda create --name conf_selection python=3.11.7`
- Install the required packages: `pip install -r requirements.txt`

## Guide to the Jupyter notebooks

### FeatureSpaceLigandSelection.ipynb

Contains code for the generation of the bisphosphine ligand space and the selection of ligands for testing in the conformer selection workflow. See Section 2 of the Supporting Information.

### ConformerEnsembleAnalysis.ipynb

Contains code used for the analysis of CREST conformer ensembles. See Section 3 of the Supporting Information. For details on how CREST conformer searches were performed, see Section 1.1 of the Supporting Information.

### CREST_DFT_FeatureComparison.ipynb

Contains code for the comparison of the electronic, steric and geometric features obtained for the DFT-refined and non-DFT-refined (geometries obtained from CREST) structures, see Section 4 of the Supporting Information.

### EquidistantConformers.ipynb

Contains code for selection of conformers based on GFN2-xTB energy as well as steric/geometric bisphosphine ligand features. See Section 5 of the Supporting Information.