import numpy
from ase import Atom, Atoms
from ase.visualize import view

from Complex_Gen.complex_structure import Ligand, Complex
from Complex_Gen.functional import ase_to_xyz, get_atoms_index, view_smiles, Center_Geo_Type


def CO():
    C1 = Atom('C', (0, 0, 0))
    O1 = Atom('O', (0, 0, 1.2))
    return Atoms([C1, O1])


def H():
    H1 = Atom('H', (0, 0, 0))
    return Atoms([H1])


def PH3():
    P1 = Atom('P', (0, 0, 0))
    H1 = Atom('H', (0, -1, 1))
    H2 = Atom('H', (0.866, 0.5, 1))
    H3 = Atom('H', (-0.866, 0.5, 1))
    return Atoms([P1, H1, H2, H3])

def CH3():
    C1 = Atom('C', (0, 0, 0))
    H1 = Atom('H', (0, -1, 1))
    H2 = Atom('H', (0.866, 0.5, 1))
    H3 = Atom('H', (-0.866, 0.5, 1))
    return Atoms([C1, H1, H2, H3])


ligand_smiles = r"CC1(OC(=C(O1)CP(C2=CC=CC=C2)C3=CC=CC=C3)CP(C4=CC=CC=C4)C5=CC=CC=C5)C"
view_smiles(ligand_smiles)

idx = get_atoms_index(smiles=ligand_smiles, atom_type="P")
print(idx)

# view_smiles(ligand_smiles)
# binding_sites_idx = get_atoms_index(smiles=ligand_smiles, atom_type="P")

ligand1 = Ligand(structure=H(), binding_sites_idx=[[0]], sites_loc_idx=[0])
ligand2 = Ligand(structure=CO(), binding_sites_idx=[[0]], sites_loc_idx=[1])
ligand3 = Ligand(smiles=ligand_smiles, binding_sites_idx=idx, sites_loc_idx=[2, 3])
# ligand3 =  Ligand(smiles="C", binding_sites_idx=[[0]], sites_loc_idx=[2])
ligand4 = Ligand(smiles="CC(=O)OC=C", binding_sites_idx=[[4,5]], sites_loc_idx=[4])

shape = Center_Geo_Type().trigonal_bipyramidal()
com = Complex(center_atom="Rh", ligands=[ligand1, ligand2, ligand3, ligand4], shape=shape)
com.generate_complex(max_attempt=1000, tol_min_dst=1.2)

view(com.complex)

# xyz = ase_to_xyz(com.complex)

# print(xyz)
