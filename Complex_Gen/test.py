import numpy
from ase import Atoms, Atom
from ase.calculators.emt import EMT
from ase.data import covalent_radii
from ase.optimize import BFGS
from ase.visualize import view
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from complex_structure import Ligand, Complex


def CO():
    C1 = Atom('C', (0, 0, 0))
    O1 = Atom('O', (0, 0, 1.2))
    anchor = numpy.array([0, 0, 0])
    return Atoms([C1, O1])


def C2H4():
    C1 = Atom('C', (-0.6, 0, 0))
    C2 = Atom('C', (0.6, 0, 0))
    H1 = Atom('H', (-1.1, 0.9, 0))
    H2 = Atom('H', (-1.1, -0.9, 0))
    H3 = Atom('H', (1.1, 0.9, 0))
    H4 = Atom('H', (1.1, -0.9, 0))
    anchor = numpy.array([0, 0, 0])
    return Atoms([C1, C2, H1, H2, H3, H4])


def H():
    H1 = Atom('H', (0, 0, 0))
    anchor = numpy.array([0, 0, 0])
    return Atoms([H1])


def PH3():
    P1 = Atom('P', (0, 0, 0))
    H1 = Atom('H', (-1, 0, 1))
    H2 = Atom('H', (0.5, 0.5 * 3 ** 0.5, 1))
    H3 = Atom('H', (0.5, -0.5 * 3 ** 0.5, 1))
    anchor = numpy.array([0, 0, 0])
    return Atoms([P1, H1, H2, H3])

def C3H6():
    C1 = Atom('C', (-0.6, 0, 0))
    C2 = Atom('C', (0.6, 0, 0))
    C3 = Atom('C', (1, 0, 1))

    anchor = numpy.array([0, 0, 0])
    return Atoms([C1, C2, C3,])

def benzene_ring():
    benzene = Atoms('C6', positions=[(0.0000, 1.4027, 0.0000),
                                        (1.2148, 0.7014, 0.0000),
                                        (1.2148, -0.7014, 0.0000),
                                        (0.0000, -1.4027, 0.0000),
                                        (-1.2148, -0.7014, 0.0000),
                                        (-1.2148, 0.7014, 0.0000)])
    anchor = numpy.array([0, 0, 0])
    return benzene, anchor

##############################################
#
ligand1 = Ligand(smiles="c1ccccc1CC=CO", binding_sites_idx=[0,1,2,3,4,5])
ligand2 = Ligand(smiles="C=CCC", binding_sites_idx=[0, 1])
ligand3 = Ligand(structure=CO(), binding_sites_idx=[0])
ligand4 = Ligand(structure=CO(), binding_sites_idx=[0])
ligand5= Ligand(structure=CO(), binding_sites_idx=[0])

# ligand2 = Ligand(*PH3(), binding_sites="P")
# ligand3 = Ligand(*CO(), binding_sites="C", direction=None)
# ligand4 = Ligand(*CO(), binding_sites="C")
# # ligand5 = Ligand(*CO(), binding_sites="C")
# ligand5 = Ligand(*benzene_ring(), binding_sites="ring")
#
shape = "pentagonal_bipyramidal"
com = Complex(center_atom="Rh", ligands=[ligand1, ligand2, ligand3, ligand4, ligand5], shape=shape)
com.generate_complex()
# #
view(com.complex)

##############################################





