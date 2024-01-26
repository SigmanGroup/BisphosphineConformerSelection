# A file which contains all of the descriptor functions necessary to run the notebooks

# These ones from the y_equidistant notebook:

import numpy as np
import pandas as pd
from morfeus import read_xyz, BiteAngle, BuriedVolume, SolidAngle
from morfeus.utils import get_connectivity_matrix

def get_bite_angle(xyz_file):
    elements, coordinates = read_xyz(xyz_file)
    m_index = np.where(elements == 'Pd')[0][0]
        
    connectivity_matrix = get_connectivity_matrix(coordinates, elements, radii_type='pyykko', 
                                                  scale_factor=1.2)
    connected_atoms = np.where(connectivity_matrix[m_index, :])[0]
    donor_atoms = []
    for i in connected_atoms:
        if elements[i] == 'P':
            donor_atoms.append(i)
            
    ba = BiteAngle(coordinates, m_index+1, donor_atoms[0]+1, donor_atoms[1]+1)
    bite_angle = ba.angle
    
    return bite_angle

def get_buried_volume(xyz_file):
    elements, coordinates = read_xyz(xyz_file)
    m_index = np.where(elements == 'Pd')[0][0]
        
    connectivity_matrix = get_connectivity_matrix(coordinates, elements, radii_type='pyykko', 
                                                  scale_factor=1.2)
    connected_atoms = np.where(connectivity_matrix[m_index, :])[0]
    cl_atoms = []
    for i in connected_atoms:
        if elements[i] == 'Cl':
            cl_atoms.append(i)
            
    bv = BuriedVolume(elements, coordinates, m_index+1, include_hs=True, radius=3.5, 
                      excluded_atoms=[cl_atoms[0]+1, cl_atoms[1]+1])
    percent_bv = bv.fraction_buried_volume * 100
    
    return percent_bv

def get_solid_angle(xyz_file):
    elements, coordinates = read_xyz(xyz_file)
            
    mask = elements != 'Cl'
    elements = elements[mask]
    coordinates = coordinates[mask]
    m_index = np.where(elements == 'Pd')[0][0] + 1
    
    sa = SolidAngle(elements, coordinates, metal_index=m_index)
    
    solid_angle = sa.solid_angle
    solid_cone_angle = sa.cone_angle
    g_param = sa.G
    
    return solid_angle, solid_cone_angle, g_param

def all_buried_volumes(xyz_file):
    elements, coordinates = read_xyz(xyz_file)
    m_index = np.where(elements == 'Pd')[0][0]
    
    connectivity_matrix = get_connectivity_matrix(coordinates, elements, radii_type='pyykko',
                                                 scale_factor=1.2)
    connected_atoms = np.where(connectivity_matrix[m_index, :])[0]
    cl_atoms = []
    for i in connected_atoms:
        if elements[i] == 'Cl':
            cl_atoms.append(i)
            
    bv3 = BuriedVolume(elements, coordinates, m_index+1, include_hs=True, radius=3,
                      excluded_atoms=[cl_atoms[0]+1, cl_atoms[1]+1])
    percent_bv3 = bv3.fraction_buried_volume * 100
    
    bv4 = BuriedVolume(elements, coordinates, m_index+1, include_hs=True, radius=4,
                      excluded_atoms=[cl_atoms[0]+1, cl_atoms[1]+1])
    percent_bv4 = bv4.fraction_buried_volume * 100
    
    bv5 = BuriedVolume(elements, coordinates, m_index+1, include_hs=True, radius=5,
                      excluded_atoms=[cl_atoms[0]+1, cl_atoms[1]+1])
    percent_bv5 = bv5.fraction_buried_volume * 100
    
    bv6 = BuriedVolume(elements, coordinates, m_index+1, include_hs=True, radius=6,
                      excluded_atoms=[cl_atoms[0]+1, cl_atoms[1]+1])
    percent_bv6 = bv4.fraction_buried_volume * 100
    
    bv7 = BuriedVolume(elements, coordinates, m_index+1, include_hs=True, radius=7,
                      excluded_atoms=[cl_atoms[0]+1, cl_atoms[1]+1])
    percent_bv7= bv7.fraction_buried_volume * 100
    
    return percent_bv3, percent_bv4, percent_bv5, percent_bv6, percent_bv7