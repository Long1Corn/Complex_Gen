import numpy
from ase import Atom, Atoms
from ase.visualize import view

from Complex_Gen.complex_structure import Ligand, Complex
from Complex_Gen.functional import ase_to_xyz


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


ligand1 = Ligand(structure=H(), binding_sites_idx=[[0]], sites_loc_idx=[0])
ligand2 = Ligand(structure=CO(), binding_sites_idx=[[0]], sites_loc_idx=[1])
ligand3 = Ligand(smiles="C1=CC=C(C3=C1C(C2=CC=CC(=C2O3)P5C4=C(C=CC=C4)OC6=CC=CC=C56)(C)C)P8C7=CC=CC=C7OC9=CC=CC=C89",
                 binding_sites_idx=[[14], [30]], sites_loc_idx=[2, 3])
# ligand3 = Ligand(smiles="c1cnc2c(c1)ccc3cccnc23", binding_sites_idx=[2, 12], sites_loc_idx=[1, 3])
# ligand4 = Ligand(structure=PH3(), binding_sites_idx=[0], sites_loc_idx=[3])
# ligand3 = Ligand(smiles=None, binding_sites_idx=[0], sites_loc_idx=[1])
# ligand4= Ligand(smiles="P", binding_sites_idx=[0], sites_loc_idx=[3])
ligand5 = Ligand(structure=CO(), binding_sites_idx=[[0]], sites_loc_idx=[4])

shape = "pentagonal_bipyramidal"
com = Complex(center_atom="Rh", ligands=[ligand1, ligand2, ligand3, ligand5], shape=shape)
com.generate_complex()

view(com.complex)

xyz = ase_to_xyz(com.complex)

print(xyz)
