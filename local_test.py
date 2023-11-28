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


ligand_smiles = r"C1=CC(=CC=C1)CCCCC2=CC=CC=C2"

view_smiles(ligand_smiles)
# binding_sites_idx = get_atoms_index(smiles=ligand_smiles, atom_type="P")

ligand1 = Ligand(structure=H(), binding_sites_idx=[[0]], sites_loc_idx=[3])
ligand2 = Ligand(structure=CO(), binding_sites_idx=[[0]], sites_loc_idx=[2])
ligand3 = Ligand(smiles=ligand_smiles, binding_sites_idx=[[0, 1, 2, 3, 4, 5], [15, 10, 11, 12, 13, 14]],
                 sites_loc_idx=[0, 1])
# ligand3 = Ligand(smiles="c1cnc2c(c1)ccc3cccnc23", binding_sites_idx=[2, 12], sites_loc_idx=[1, 3])
# ligand4 = Ligand(structure=PH3(), binding_sites_idx=[0], sites_loc_idx=[3])
# ligand3 = Ligand(smiles=None, binding_sites_idx=[0], sites_loc_idx=[1])
# ligand4= Ligand(smiles="P", binding_sites_idx=[0], sites_loc_idx=[3])
ligand5 = Ligand(structure=CO(), binding_sites_idx=[[0]], sites_loc_idx=[4])
# ligand5 = Ligand(smiles="CC(=O)OC=C", binding_sites_idx=[[4, 5]], sites_loc_idx=[4])

shape = Center_Geo_Type().trigonal_bipyramidal()
com = Complex(center_atom="Rh", ligands=[ ligand2, ligand3, ligand5], shape=shape)
com.generate_complex(max_attempt=1000)

view(com.complex)

# xyz = ase_to_xyz(com.complex)

# print(xyz)
