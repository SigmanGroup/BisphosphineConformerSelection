"""
General utility functions for the project.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def count_conformer_number(path: Path, ligand: str) -> int:
    """
    Counts the number of conformers for a given ligand in a directory.
    
    Args:
        path (Path): Path to directory.
        ligand (str): Ligand name.
        
    Returns:
        int: Number of conformers.
    """
    
    all_files = list(path.rglob('*'))
    
    count_list = [i for i in all_files if ligand in i.name]
    count = len(count_list)

    return count


def get_xtb_energy(xyz_file: str) -> float:
    """
    Gets the xTB energy from a xyz file. This is the second line of the xyz file.
    
    Args:
        xyz_file (str): Path to xyz file.
    
    Returns:
        float: xTB energy.
    """
    
    with open(xyz_file, "r") as file:
        energy = float(file.readlines()[1])

    return energy


def select_equidistant_values(df: pd.DataFrame, column: str, y: int) -> pd.DataFrame:
    """
    Selects equidistant values from a dataframe.
    The values are selected based on the minimum and maximum values of a given column.
    
    Args:
        df (pd.DataFrame): Dataframe.
        column (str): Column name.
        y (int): Number of values to select.
        
    Returns:
        pd.DataFrame: Dataframe with selected values.
    """

    sorted_df = df.sort_values(column)
    indices = np.linspace(0, len(sorted_df) - 1, y, dtype=int)
    selected_values = sorted_df.iloc[indices]

    return selected_values


def select_lec(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Selects the lowest energy conformer from a dataframe.
    
    Args:
        df (pd.DataFrame): Dataframe.
        column (str): Column name.
        
    Returns:
        pd.DataFrame: Dataframe with selected conformer.
    """

    lec_df = df.sort_values(column)
    indices = np.linspace(0, len(lec_df) - 1, 1, dtype=int)
    lec = lec_df.iloc[indices]

    return lec


def read_energy_selection_txt_file(txt_file: str, sele_num: int = 10) -> list:
    """
    Reads the selection.txt file from conformers selected using xTB energy.
    Returns a list of the selected conformers.
    
    Args:
        txt_file (str): Path to selection.txt file.
        sele_num (int, optional): Number of conformers selected. Defaults to 10.
        
    Returns:
        list: List of selected conformers.
    """
    
    selections = []

    with open(txt_file, 'r') as file:
        lines = []
        for line in file.readlines():
            lines.append(line.strip())

        energy_str = "energy"
        for row in lines:
            if row.find(energy_str) != -1:
                idx = lines.index(row)
                selections.append(lines[idx+1:idx+sele_num+2])

    dft = []
    for selection in selections[0]:
        file = str(selection)
        if file in dft:
            continue
        else:
            dft.append(file)

    return dft


def read_selection_txt_file(txt_file: str, sele_num: int = 10) -> tuple[list, list]:
    """
    Reads the selection.txt file from conformers selected using MORFEUS features of xtb energy.
    Returns a list of the selected conformers.
    
    Args:
        txt_file (str): Path to selection.txt file.
        sele_num (int, optional): Number of conformers selected. Defaults to 10.
        
    Returns:
        list: List of selected conformers.
    """
    
    bite_angle_selections = []
    buried_volume_selections = []

    with open(txt_file, 'r') as file:
        lines = []
        for line in file.readlines():
            lines.append(line.strip())

        bite_str = "bite_angle"
        for row in lines:
            if row.find(bite_str) != -1:
                idx = lines.index(row)
                bite_angle_selections.append(lines[idx+1:idx+sele_num+2])

        vbur_str = "buried_volume"
        for row in lines:
            if row.find(vbur_str) != -1:
                idx = lines.index(row)
                buried_volume_selections.append(lines[idx+1:idx+sele_num+2])

    bite_angle_dft = []
    buried_volume_dft = []

    for selection in bite_angle_selections[0]:
        file = str(selection)
        if 'LEC:' in selection:
            file = selection.strip('LEC: ')
            if file in bite_angle_dft:
                continue
            else:
                bite_angle_dft.append(file)
        else:
            if file in bite_angle_dft:
                continue
            else:
                bite_angle_dft.append(file)

    for selection in buried_volume_selections[0]:
        file = str(selection)
        if 'LEC:' in selection:
            file = selection.strip('LEC: ')
            if file in buried_volume_dft:
                continue
            else:
                buried_volume_dft.append(file)
        else:
            if file in buried_volume_dft:
                continue
            else:
                buried_volume_dft.append(file)

    return bite_angle_dft, buried_volume_dft


def percent_difference(old_val: float, new_val: float) -> float:
    """
    Calculates the percent difference between two values.
    
    Args:
        old_val (float): Old value.
        new_val (float): New value.
    
    Returns:
        float: Percent difference.
    """
    
    diff = abs(((new_val - old_val) / old_val) * 100.0)
    return diff


def difference(old_val: float, new_val: float) -> float:
    """
    Calculates the difference between two values.
    
    Args:
        old_val (float): Old value.
        new_val (float): New value.
        
    Returns:
        float: Difference.
    """

    diff = abs(new_val - old_val)
    return diff
