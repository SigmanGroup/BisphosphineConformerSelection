from pathlib import Path

def count_conformer_number(path: Path, ligand):
    all_files = list(path.rglob('*'))
    
    count_list = [i for i in all_files if ligand in i.name]
    count = len(count_list)

    return count