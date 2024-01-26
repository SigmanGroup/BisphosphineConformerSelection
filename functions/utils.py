# A file which contains all of the utility functions necessary to run the notebooks

# These ones from the y_equidistant notebook:

import numpy as np
import pandas as pd

# TODO: probably should move this to the conformers python file

def get_xtb_energy(xyz_file):
    with open(xyz_file, "r") as file:
        energy = float(file.readlines()[1])

    return energy

def select_equidistant_values(df, column, y):
    sorted_df = df.sort_values(column)
    min_value = sorted_df[column].min()
    max_value = sorted_df[column].max()

    indices = np.linspace(0, len(sorted_df) - 1, y, dtype=int)
    selected_values = sorted_df.iloc[indices]

    return selected_values

def select_lec(df, column):
    lec_df = df.sort_values(column)
    indices = np.linspace(0, len(lec_df) - 1, 1, dtype=int)
    lec = lec_df.iloc[indices]

    return lec

def read_selection_txt_file(txt_file, sele_num=10):
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

# this is from the DFT property analysis section

def percent_difference(old_val, new_val):
    diff = abs(((new_val - old_val) / old_val) * 100.0)
    return diff

def difference(old_val, new_val):
    diff = abs(new_val - old_val)
    return diff