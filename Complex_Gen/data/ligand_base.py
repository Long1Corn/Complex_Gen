""" this file contains the information of a ligand base """
from ase import Atom, Atoms
from rdkit import Chem
from rdkit.Chem import AllChem


# structure, binding_sites_idx, sites_loc_idx
def CO():
    C1 = Atom('C', (0, 0, 0))
    O1 = Atom('O', (0, 0, 1.2))
    return Atoms([C1, O1]), [0], [0]


def H():
    H1 = Atom('H', (0, 0, 0))
    return Atoms([H1]), [0], [0]











