import numpy
from ase import Atom, Atoms
from ase.visualize import view

from Complex_Gen.complex_structure import Ligand, Complex


def CO():
    C1 = Atom('C', (0, 0, 0))
    O1 = Atom('O', (0, 0, 1.2))
    return Atoms([C1, O1])


ligand1 = Ligand(smiles="c1ccccc1CC=CO", binding_sites_idx=[0,1,2,3,4,5])
ligand2 = Ligand(smiles="C=CCC", binding_sites_idx=[0, 1])
ligand3 = Ligand(structure=CO(), binding_sites_idx=[0])
ligand4 = Ligand(structure=CO(), binding_sites_idx=[0])
ligand5= Ligand(structure=CO(), binding_sites_idx=[0])

shape = "pentagonal_bipyramidal"
com = Complex(center_atom="Rh", ligands=[ligand1, ligand2, ligand3, ligand4, ligand5], shape=shape)
com.generate_complex()

view(com.complex)