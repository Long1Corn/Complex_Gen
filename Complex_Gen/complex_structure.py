import random

import numpy as np
from ase import Atoms, Atom
from rdkit import Chem
from rdkit.Chem import AllChem

from Complex_Gen.functional import get_bond_dst, find_ligand_pos, rodrigues_rotation_matrix, \
    rotate_bidendate_angel, rotate_point_about_vector, check_atoms_distance, Center_Geo_Type


class Ligand:
    """
    A class to represent a ligand.
    """

    def __init__(self, binding_sites_idx: [[int]], sites_loc_idx: [int], smiles: str = None, structure: Atoms = None):
        """
        :param smiles: SMILES string of the ligand
        :param structure: ASE Atoms object of the ligand (provide either one of the two)
        :param binding_sites_idx: A list of atom indices of the binding sites
        :param sites_loc_idx: A list of atom indices of the sites location
        :param anchor: position of the binding site within the ligand
        :param direction: direction of the ligand, default to be the geometric center of the ligand
        """
        self._structure = None
        self._rdkit_mol = None
        self._binding_sites_idx = binding_sites_idx
        self._sites_loc_idx = sites_loc_idx
        self._smiles = smiles
        self._structure = structure
        self._rdkit_mol = None

        if len(self._sites_loc_idx) == 1:
            self.dentate = 1
        elif len(self._sites_loc_idx) == 2:
            self.dentate = 2

        self._gen_conformer()

    def _gen_conformer(self, max_conformers=200):

        # get ligand structure (ASE ATOMS) from smiles or structure
        if self._smiles is not None:
            self._get_structure_from_smiles(max_conformers=max_conformers)

        # get binding sites
        if self.dentate == 1:  # mono-dentate
            if len(self._binding_sites_idx[0]) == 1:
                self._binding_sites = self._structure[self._binding_sites_idx[0][0]].symbol
            elif len(self._binding_sites_idx[0]) >= 2:
                self._binding_sites = "="

        elif self.dentate == 2:  # bi-dentate
            # todo: add support for pi bonding sites for bi-dentate ligands
            self._binding_sites = []

            if len(self._binding_sites_idx[0]) == 1:
                self._binding_sites.append(self._structure[self._binding_sites_idx[0][0]].symbol)
            else:
                self._binding_sites.append("=")

            if len(self._binding_sites_idx[1]) == 1:
                self._binding_sites.append(self._structure[self._binding_sites_idx[1][0]].symbol)
            else:
                self._binding_sites.append("=")

        else:
            raise ValueError("Only mono-dentate and bi-dentate ligands are supported")

        # get anchor and direction

        self._anchor = self._find_anchor(self.dentate)
        self._direction = self._find_ligand_pos()

    def _get_structure_from_smiles(self, max_conformers=200, mirror=False):
        # Create RDKit molecule from SMILES

        if self._rdkit_mol is None:
            mol = Chem.MolFromSmiles(self._smiles)
            # Add hydrogens
            mol = Chem.AddHs(mol)
            # Generate 3D coordinates
            # AllChem.EmbedMolecule(mol, AllChem.ETKDG(), randomSeed=random.randint(0, 10000))
            AllChem.EmbedMultipleConfs(mol, numConfs=max_conformers, params=AllChem.ETKDG())
            self._rdkit_mol = mol

        # Extract atomic numbers
        atomic_numbers = [atom.GetAtomicNum() for atom in self._rdkit_mol.GetAtoms()]

        # Get a random conformer
        conformer = self._rdkit_mol.GetConformer(random.randint(0, max_conformers - 1))

        # Extract coordinates
        coords = [conformer.GetAtomPosition(i) for i in range(self._rdkit_mol.GetNumAtoms())]
        coords = np.array([[pos.x, pos.y, pos.z] for pos in coords])

        # Create an ASE Atoms object
        ase_atoms = Atoms(numbers=atomic_numbers, positions=coords)

        if mirror:
            # 50% chance to mirror the ligand
            if random.random() > 0.5:
                ase_atoms.positions[:, 0] = -ase_atoms.positions[:, 0]
        #
        # randomly rotate the ligand
        ase_atoms.rotate(random.random() * 360, 'z')

        self._structure = ase_atoms

    def _find_anchor(self, dentate: int) -> np.ndarray:
        if dentate == 1:
            # find the binding site to be the geometric center of all binding atoms
            anchor = np.mean(self._structure.get_positions()[self._binding_sites_idx[0]], axis=0)

        elif dentate == 2:
            # find the bindsite to be the locations of each binding sites
            anchor = np.array([np.mean(self._structure.get_positions()[binding_sites_idx], axis=0)
                               for binding_sites_idx in self._binding_sites_idx])

        return anchor

    def _find_ligand_pos(self) -> np.ndarray:
        """try to find the name of the binding site and the geometric center of the ligand"""
        ligand_pos = find_ligand_pos(self._structure, self._anchor, self._binding_sites, self.dentate)

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

    def __init__(self, center_atom: str, shape: np.ndarray, ligands: [Ligand]):
        """
        :param center_atom: atom symbol of the center metal atom
        :param shape: coordination geometry of the center metal atom "pentagonal_bipyramidal", "octahedral"
        :param ligand: A list of ligand objects
        """
        self._center_atom = Atom(center_atom, (0, 0, 0))
        self._shape = shape
        self._ligands = ligands
        self.complex = None

    def generate_complex(self, max_attempt=200, tol_min_dst=1.0) -> Atoms or None:
        """
        Generate the initial complex structure.
        :param max_attempt: maximum number of attempts to generate the complex, also control number of conformers
        :return: complex structure
        """

        center_geo = self._shape

        com_list = []
        dst_list = []

        for attempt in range(max_attempt):

            com = Atoms([self._center_atom])
            bond_dst_list = []
            ligand_coord_list = []

            for i in range(len(self._ligands)):

                self._ligands[i]._gen_conformer()

                num_dentate = self._ligands[i].dentate

                # get angel and direction of the binding site
                if num_dentate == 1:
                    angel_factor = 1.0
                    direction = center_geo[self._ligands[i]._sites_loc_idx[0]]

                elif num_dentate == 2:
                    # get angel between two binding sites
                    v1 = center_geo[self._ligands[i]._sites_loc_idx[0]]
                    v2 = center_geo[self._ligands[i]._sites_loc_idx[1]]
                    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-2)
                    theta_rad = np.arccos(cos_theta)
                    angel_factor = np.cos(theta_rad / 2)

                    direction = [v1, v2]

                # get bond distance
                bond_dst = get_bond_dst(self._center_atom.symbol, self._ligands[i]._binding_sites,
                                        num_dentate=num_dentate, )
                bond_dst_list.append(bond_dst)

                # get ligand position and combine the ligand
                ligand_coord = self.place_ligand(self._ligands[i], direction, bond_dst * angel_factor)
                com = com + ligand_coord
                ligand_coord_list.append(ligand_coord)

            # check if the ligands are too close to each other
            min_dst, min_dst_center = check_atoms_distance(com, ligand_coord_list)
            if min_dst_center > min(bond_dst_list) - 1e-3:
                com_list.append(com)
                dst_list.append(min_dst)

        if len(com_list) == 0:
            self.complex = None
            print(f"Failed to generate complex after {max_attempt} attempts")
            print(f"Cannot find a good geometry given current ligand")
            return self.complex

        # get the max min_dst and idx
        max_min_dst = max(dst_list)
        idx = dst_list.index(max_min_dst)

        # todo: there should be a better way to handle this
        if max_min_dst > tol_min_dst:

            self.complex = com_list[idx]
        else:
            self.complex = None
            print(f"Failed to generate complex after {max_attempt} attempts")
            print(f"Maximum distance between ligands is {max_min_dst}")

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

        # get position of anchor and directional vector of the binding sites
        if ligand.dentate == 1:
            anchor = ligand._anchor

        elif ligand.dentate == 2:
            anchor = np.mean(ligand._anchor, axis=0)
            v1 = pos[0]
            v2 = pos[1]
            pos = (v1 + v2) / 2

            if np.linalg.norm(pos) == 0:
                pos = np.cross(v1 + 1e-3 * np.random.randn(1), v2 + 1e-3 * np.random.randn(1))

        pos = pos / np.linalg.norm(pos)

        # get ligand original position vector
        ligand_pos = ligand._direction

        # align the original position vector and directional vector by rotating ligand
        R = rodrigues_rotation_matrix(ligand_pos, pos)
        for atom in ligand_structure:
            atom.position = R @ (atom.position - anchor) + pos * bond_dst

        # for bidentate, rotato the ligand around the directional vector to match two binding sites
        if ligand.dentate == 2:
            # rotate the ligand around pos to minimize direction between two binding sites
            rotate_angel = rotate_bidendate_angel(
                np.mean(ligand_structure.positions[ligand._binding_sites_idx[0]], axis=0),
                np.mean(ligand_structure.positions[ligand._binding_sites_idx[1]], axis=0),
                v1, v2, pos)

            # rotate the ligand around pos
            for atom in ligand_structure:
                atom.position = rotate_point_about_vector(atom.position, pos, rotate_angel)

        return ligand_structure

    def __repr__(self):
        return f"Complex:{self._center_atom.symbol}{self._ligands}"
