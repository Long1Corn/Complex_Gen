import numpy as np
from ase import Atoms
from ase.data import covalent_radii


def unit_vector(vector: np.ndarray):
    """ Returns the unit vector of the vector. """
    return vector / np.linalg.norm(vector, axis=-1)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'. """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def rodrigues_rotation_matrix(a, b):
    """ Returns the rotation matrix that aligns vector 'a' with vector 'b'. """
    # Normalize the input vectors
    a = unit_vector(a)
    b = unit_vector(b)

    # Calculate the rotation axis via cross product
    axis = np.cross(a, b)
    axis_len = np.linalg.norm(axis)

    # If vectors are parallel or anti-parallel, return identity or inversion
    if axis_len < 1E-7:
        return np.eye(3) if np.dot(a, b) > 0 else -np.eye(3)

    # Normalize the rotation axis
    axis /= axis_len

    # Calculate the rotation angle
    angle = angle_between(a, b)

    # Rodrigues' rotation formula
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    rotation_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K

    return rotation_matrix


def find_mol_center(structure: Atoms) -> np.ndarray:
    # find geometric center of a molecule (ase atom)
    center = np.array([0, 0, 0])
    for atom in structure:
        center = center + atom.position
    center = center / len(structure)
    return center


def find_near_atoms(structure: Atoms, anchor: np.ndarray, num: int):
    # find the index of nearest N atoms to the anchor in a molecule (ase atom)
    dst = []
    for atom in structure:
        dst.append(np.linalg.norm(atom.position - anchor))
    dst = np.array(dst)
    idx = np.argsort(dst)
    return idx[:num]


def find_ligand_pos(structure, anchor, site: str) -> np.ndarray:
    """
    Find the directional vector of a ligand using anchor and the geometric center of the ligand.
    :param structure: ligand structure
    :param anchor: 3d position within the ligand
    :return: 3d directional vector of the ligand
    """
    if len(structure) == 1:  # ligand is single atom
        ligand_pos = np.array([0, 0, 1])

    elif site == "=":  # bind to pi bond
        near_atoms_idx = find_near_atoms(structure, anchor, 2)
        # find the vector of the pi bond
        v_pi = structure[near_atoms_idx[0]].position - structure[near_atoms_idx[1]].position

        # find the vector of the ligand
        ligand_center = find_mol_center(structure)
        v_ligand = ligand_center - anchor

        # find the normal component of the ligand vector subtracted by the pi bond vector
        v_normal = v_ligand - np.dot(v_ligand, v_pi) * v_pi / np.linalg.norm(v_pi) **2

        ligand_pos = v_normal

    elif site == "ring":  # bind to ring
        near_atoms_idx = find_near_atoms(structure, anchor, 3)

        v_1 = structure[near_atoms_idx[0]].position - structure[near_atoms_idx[1]].position
        v_2 = structure[near_atoms_idx[0]].position - structure[near_atoms_idx[2]].position

        # find the normal vector of the ring plane described by v1, v2
        v_normal = np.cross(v_1, v_2)

        # make v_normal and ligand the same direction
        ligand_center = find_mol_center(structure)
        v_ligand = ligand_center - anchor

        v_normal = v_normal * (np.sign(np.dot(v_normal, v_ligand)) + 1e-2)

        ligand_pos = v_normal
    else:  # bind to atom
        ligand_center = find_mol_center(structure)
        ligand_pos = ligand_center - anchor

    return ligand_pos


def get_bond_dst(atom1: str, atom2: str) -> float:
    # todo: implement pi bond and ring
    """
    Get the bond distance between two atoms.
    set to be 1.1 times of the sum of covalent radii of the two atoms.

    :param atom1: atom symbol
    :param atom2: atom symbol
    :return: bond distance
    """

    if atom1 == "=" or atom2 == "=":
        # raise NotImplementedError("Bond distance for pi bond not implemented yet.")
        if atom1 == "=":
            atom = atom2
        else:
            atom = atom1

        s1 = Atoms(atom).numbers[0]
        dst = covalent_radii[s1] + 0.7  # assuming bond length of pi site is 0.6 A

    elif atom1 == "ring" or atom2 == "ring":
        # raise NotImplementedError("Bond distance for ring not implemented yet.")
        if atom1 == "ring":
            atom = atom2
        else:
            atom = atom1

        s1 = Atoms(atom).numbers[0]
        dst = covalent_radii[s1] + 0.6  # assuming bond length of pi site is 0.5 A
    else:
        s1 = Atoms(atom1).numbers[0]
        s2 = Atoms(atom2).numbers[0]
        dst = covalent_radii[s1] + covalent_radii[s2]
    return dst


def get_center_geo(geo_type: str) -> np.ndarray:
    geo_type_dict = {"pentagonal_bipyramidal": [[0, 0, 1],
                                                [0, 0, -1],
                                                [1, 0, 0],
                                                [-0.5, 0.5 * 3 ** 0.5, 0],
                                                [-0.5, -0.5 * 3 ** 0.5, 0]],
                     "octahedral": [[0, 0, 1],
                                    [0, 0, -1],
                                    [1, 0, 0],
                                    [0, 1, 0],
                                    [-1, 0, 0],
                                    [0, -1, 0]], }
    if geo_type not in geo_type_dict.keys():
        raise ValueError(f"Geometry type {geo_type} not found in geo_type_dict")
    else:
        center_geo = np.array(geo_type_dict[geo_type])

    center_geo = center_geo / np.linalg.norm(center_geo, axis=-1)[:, np.newaxis]
    return center_geo
