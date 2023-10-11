import numpy as np
from ase import Atoms
from ase.data import covalent_radii
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R


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


def find_ligand_pos(structure, anchor, site, sites_loc_idx, center_geo_type=None) -> np.ndarray:
    """
    Find the directional vector of a ligand using anchor and the geometric center of the ligand.
    :param structure: ligand structure
    :param anchor: 3d position within the ligand, could be np.ndarray or [np.ndarray]
    :param site: type binding site of the ligand, could be str or [str]
    :param sites_loc_idx: index of the binding site location, only used for bi-dentate
    :param center_geo_type: type of geometric center of the ligand, only used for bi-dentate
    :return: 3d directional vector of the ligand
    """
    if len(structure) == 1:  # ligand is single atom
        ligand_pos = np.array([0, 0, 1])

    # elif site == "=":  # bind to pi bond
    #     near_atoms_idx = find_near_atoms(structure, anchor, 2)
    #     # find the vector of the pi bond
    #     v_pi = structure[near_atoms_idx[0]].position - structure[near_atoms_idx[1]].position
    #
    #     # find the vector of the ligand
    #     ligand_center = find_mol_center(structure)
    #     v_ligand = ligand_center - anchor
    #
    #     # find the normal component of the ligand vector subtracted by the pi bond vector
    #     v_normal = v_ligand - np.dot(v_ligand, v_pi) * v_pi / np.linalg.norm(v_pi) **2
    #
    #     ligand_pos = v_normal

    elif site == "ring" or site == "=":  # bind to ring
        # find the plane of the ring or pi bond
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

    elif len(site) == 2: # bind to bi-dentate
        geo_vector = get_center_geo(center_geo_type)

        geo_vector = geo_vector[sites_loc_idx]

        # find the vector on plane (v1,v2) and perpendicular to v_a12
        anchor1 = anchor[0]
        anchor2 = anchor[1]
        anchors_center = (anchor1 + anchor2) / 2

        v1 = geo_vector[0]
        v2 = geo_vector[1]

        v_a12 = anchor1 - anchor2

        # Find the normal vector to the plane
        n = np.cross(v1, v2)

        # Find the desired vector
        u = np.cross(n, v_a12)

        # make v_normal and ligand the same direction
        ligand_center = find_mol_center(structure)
        v_ligand = ligand_center - anchors_center

        u = u * (np.sign(np.dot(u, v_ligand)) + 1e-2)

        ligand_pos = u

    else:  # bind to one atom site
        ligand_center = find_mol_center(structure)
        ligand_pos = ligand_center - anchor

    return ligand_pos


def get_bond_dst(atom1: str, atom2, num_dentate:int, angel_factor=None) -> float:
    # todo: implement pi bond and ring
    """
    Get the bond distance between two atoms.
    set to be 1.1 times of the sum of covalent radii of the two atoms.

    :param atom1: atom symbol
    :param atom2: atom symbol
    :return: bond distance
    """

    if num_dentate == 1 : # mono-dentate
        if atom1 == "=" or atom2 == "=":
            # raise NotImplementedError("Bond distance for pi bond not implemented yet.")
            if atom1 == "=":
                atom = atom2
            else:
                atom = atom1

            s1 = Atoms(atom).numbers[0]
            dst = covalent_radii[s1] + 0.7  # assuming bond length of pi site is 0.6 A

        elif atom1 == "ring" or atom2 == "ring":
            if atom1 == "ring":
                atom = atom2
            else:
                atom = atom1

            s1 = Atoms(atom).numbers[0]
            dst = covalent_radii[s1] + 0.6  # assuming bond length of pi site is 0.5 A
        else:
            s1 = Atoms(atom1).numbers[0]
            s2 = Atoms(atom2).numbers[0]
            dst = (covalent_radii[s1] + covalent_radii[s2]) * 1.1

    elif num_dentate == 2: # bi-dentate
        s1 = Atoms(atom1).numbers[0]
        s21 = Atoms(atom2[0]).numbers[0]
        s22 = Atoms(atom2[1]).numbers[0]

        dst = (covalent_radii[s1] + 0.5*(covalent_radii[s21] + covalent_radii[s22])) * angel_factor * 1.1

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


def rotate_point_about_vector(point, axis, angle):
    """Rotates a point about a given axis by a specified angle."""
    rotation = R.from_rotvec(axis / np.linalg.norm(axis) * angle)
    return rotation.apply(point)

def cost_function(angle, x1, x2, v1, v2, v0):
    """Compute the cost for a given rotation angle."""
    x1_rot = rotate_point_about_vector(x1, v0, angle)
    x2_rot = rotate_point_about_vector(x2, v0, angle)
    distance1 = np.linalg.norm(x1_rot - v1)
    distance2 = np.linalg.norm(x2_rot - v2)
    return distance1 + distance2

def rotate_bidendate_angel(x1, x2, v1, v2, v0):
    result = minimize(lambda angle: cost_function(angle[0], x1, x2, v1, v2, v0), [0])

    optimal_angle = result.x[0]

    return optimal_angle
