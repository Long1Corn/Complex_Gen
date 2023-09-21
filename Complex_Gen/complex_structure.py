import numpy as np
from ase import Atoms, Atom
from rdkit import Chem
from rdkit.Chem import AllChem

from Complex_Gen.functional import get_center_geo, get_bond_dst, find_ligand_pos, rodrigues_rotation_matrix


class Ligand:
    """
    A class to represent a ligand.
    """

    def __init__(self, binding_sites_idx: [int], smiles: str = None, structure: Atoms=None, anchor: np.ndarray = None, direction: np.ndarray = None):
        """
        :param smiles: SMILES string of the ligand
        :param structure: ASE Atoms object of the ligand (provide one of the two)
        :param binding_sites_idx: A list of atom indices of the binding sites
        :param anchor: position of the binding site within the ligand
        :param direction: direction of the ligand, default to be the geometric center of the ligand
        """
        self._structure = None
        self._rdkit_mol = None
        self._binding_sites_idx = binding_sites_idx

        # get ligand structure (ASE ATOMS) from smiles or structure
        if smiles is not None:
            self._smiles = smiles
            self._get_structure_from_smiles()
        elif structure is not None:
            self._structure = structure

        # get binding sites
        if len(self._binding_sites_idx) == 1:
            self._binding_sites = self._structure[self._binding_sites_idx[0]].symbol
        elif len(self._binding_sites_idx) == 2:
            self._binding_sites = "="
        else:
            self._binding_sites = "ring"

        # get anchor and direction
        if anchor is None:
            self._anchor = self._find_anchor()
        else:
            self._anchor = anchor

        if direction is None:
            self._direction = self._find_ligand_pos()
        else:
            self._direction = direction

    def _get_structure_from_smiles(self):
        # Create RDKit molecule from SMILES
        mol = Chem.MolFromSmiles(self._smiles)

        # Add hydrogens
        mol = Chem.AddHs(mol)

        # Generate 3D coordinates
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())

        # Extract atomic numbers
        atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]

        # Get a conformer (assuming only one conformer)
        conformer = mol.GetConformer()

        # Extract coordinates
        coords = [conformer.GetAtomPosition(i) for i in range(mol.GetNumAtoms())]
        coords = np.array([[pos.x, pos.y, pos.z] for pos in coords])

        # Create an ASE Atoms object
        ase_atoms = Atoms(numbers=atomic_numbers, positions=coords)

        self._structure = ase_atoms
        self._rdkit_mol = mol

    def _find_anchor(self):
        """find the binding site to be the geometric center of all binding atoms"""

        anchor = np.mean(self._structure.get_positions()[self._binding_sites_idx], axis=0)
        return anchor

    def _find_ligand_pos(self):
        """try to find the name of the binding site and the geometric center of the ligand"""
        ligand_pos = find_ligand_pos(self._structure, self._anchor, self._binding_sites)

        return ligand_pos

    @property
    def formula(self):
        return self._structure.get_chemical_formula()

    def __repr__(self):
        return f"Ligand({self.formula})"




class Complex:
    """
    A class to represent a complex structure.
    """

    def __init__(self, center_atom: str, shape: str, ligands: [Ligand]):
        """
        :param center_atom: atom symbol of the center metal atom
        :param shape: coordination geometry of the center metal atom "pentagonal_bipyramidal", "octahedral"
        :param ligand: A list of ligand objects
        """
        self._center_atom = Atom(center_atom, (0, 0, 0))
        self._shape = shape
        self._ligands = ligands
        self.complex = None

    def generate_complex(self):
        """
        Generate the initial complex structure.
        :return: complex structure
        """

        com = Atoms([self._center_atom])

        center_geo = get_center_geo(self._shape)

        if not len(center_geo) == len(self._ligands):
            raise AssertionError(f"Number of ligands ({len(self._ligands)}) does not match the shape ({self._shape})")

        for i in range(len(center_geo)):
            bond_dst = get_bond_dst(self._center_atom.symbol, self._ligands[i]._binding_sites)
            ligand_coord = self.place_ligand(self._ligands[i], center_geo[i], bond_dst)
            com = com + ligand_coord

        self.complex = com

        return self.complex

    @staticmethod
    def place_ligand(ligand: Ligand, pos, bond_dst) -> Atoms:
        """
        Place a ligand at a given position of complex.
        :param ligand: ligand object
        :param pos: reference unit direction vector of the complex bonding
        :param bond_dst: distance between the ligand and the center atom
        :return: ASE Atoms Object, ligand structure with updated positions
        """
        ligand_structure = ligand._structure.copy()
        anchor = ligand._anchor

        ligand_pos = ligand._direction

        R = rodrigues_rotation_matrix(ligand_pos, pos)
        for atom in ligand_structure:
            atom.position = R @ (atom.position - anchor) + pos * bond_dst

        return ligand_structure

    def __repr__(self):
        return f"Complex:{self._center_atom.symbol}{self._ligands}"
