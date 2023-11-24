import numpy as np
from ase import Atoms

from ase.data import covalent_radii
from matplotlib import pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdDepictor
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
    # find the index of nearest N atoms to the anchor in a molecule (ase atom) excluding hydrogen atoms
    dst = []
    for idx, atom in enumerate(structure):
        # Skip hydrogen atoms
        if atom.symbol != 'H':
            dst.append((np.linalg.norm(atom.position - anchor), idx))

    # Sort by distance, but keep the atom index information
    sorted_dst = sorted(dst, key=lambda x: x[0])

    # Extract atom indices of the closest atoms
    nearest_atom_indices = [x[1] for x in sorted_dst[:num]]

    return np.array(nearest_atom_indices)


def find_ligand_pos(structure, anchor, site, dentate) -> np.ndarray:
    """
    Find the directional vector of a ligand using anchor and the geometric center of the ligand.
    :param structure: ligand structure
    :param anchor: 3d position within the ligand, could be np.ndarray or [np.ndarray]
    :param site: type binding site of the ligand, could be str or [str]
    :param dentate:
    :return: 3d directional vector of the ligand
    """
    if dentate == 1:  # mono-dentate

        if len(structure) == 1:  # ligand is single atom
            ligand_pos = np.array([0, 0, 1])

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
        else:  # bind to one atom site
            ligand_center = find_mol_center(structure)
            ligand_pos = ligand_center - anchor

    elif dentate == 2:  # bind to bi-dentate
        # find the vector on plane (v1,v2) and perpendicular to v_a12
        anchor1 = anchor[0]
        anchor2 = anchor[1]
        anchors_center = (anchor1 + anchor2) / 2

        near_atoms_idx1 = find_near_atoms(structure, anchor1, 2)
        near_atoms_idx2 = find_near_atoms(structure, anchor2, 2)

        v1 = structure[near_atoms_idx1[1]].position - anchor1
        v2 = structure[near_atoms_idx2[1]].position - anchor2

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

    return ligand_pos


def get_bond_dst(atom1: str, atom2, num_dentate: int, angel_factor=None) -> float:
    # todo: implement pi bond and ring
    """
    Get the bond distance between two atoms.
    set to be 1.1 times of the sum of covalent radii of the two atoms.

    :param atom1: atom symbol
    :param atom2: atom symbol
    :return: bond distance
    """

    if num_dentate == 1:  # mono-dentate
        if atom1 == "=" or atom2 == "=":
            # raise NotImplementedError("Bond distance for pi bond not implemented yet.")
            if atom1 == "=":
                atom = atom2
            else:
                atom = atom1

            s1 = Atoms(atom).numbers[0]
            dst = covalent_radii[s1] + 0.6  # assuming bond length of pi site is 0.6 A

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
            dst = (covalent_radii[s1] + covalent_radii[s2]) * 1.0

    elif num_dentate == 2:  # bi-dentate
        s1 = Atoms(atom1).numbers[0]
        s21 = Atoms(atom2[0]).numbers[0]
        s22 = Atoms(atom2[1]).numbers[0]

        dst = (covalent_radii[s1] + 0.5 * (covalent_radii[s21] + covalent_radii[s22])) * angel_factor * 1.0

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

    # get angel between v and x
    angle1 = np.arccos(np.dot(v1, x1_rot) / (np.linalg.norm(v1) * np.linalg.norm(x1_rot)))
    angle2 = np.arccos(np.dot(v2, x2_rot) / (np.linalg.norm(v2) * np.linalg.norm(x2_rot)))

    return angle1 + angle2


def rotate_bidendate_angel(x1, x2, v1, v2, v0):
    starting_points = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    results = []

    for sp in starting_points:
        res = minimize(lambda angle: cost_function(angle[0], x1, x2, v1, v2, v0), [sp])
        results.append((res.fun, res.x[0]))

    # Find the best result
    best_result = min(results, key=lambda x: x[0])
    best_cost, best_angle = best_result

    return best_angle


def ase_to_xyz(atoms, decimals=8):
    """Convert an ASE Atoms object to an XYZ string with specified decimals."""
    n_atoms = len(atoms)
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()

    xyz_string = str(n_atoms) + '\n\n'  # number of atoms, then a blank comment line

    format_string = "{0} {1:." + str(decimals) + "f} {2:." + str(decimals) + "f} {3:." + str(decimals) + "f}\n"
    for symbol, position in zip(symbols, positions):
        xyz_string += format_string.format(symbol, position[0], position[1], position[2])

    return xyz_string


def view_smiles(smiles: str):
    """Visualize a molecule from a SMILES string for identifying atom indices."""
    # Convert SMILES to molecule
    mol = Chem.MolFromSmiles(smiles)

    # If RDKit was unable to parse the SMILES, return
    if mol is None:
        print("Invalid SMILES string:", smiles)
        return

    # Add hydrogens
    mol = Chem.AddHs(mol)

    # Compute 2D coordinates for visualization
    rdDepictor.Compute2DCoords(mol)

    # Get the drawing
    size = 600
    img = Draw.MolToImage(mol, size=(size, size), kekulize=True, wedgeBonds=True, bgcolor=None)
    fig, ax = plt.subplots(figsize=(6, 6))
    # plt.xlim(0,size)
    # plt.ylim(0,size)
    ax.imshow(img)
    # ax.axis("off")

    # Get molecule conformer
    conf = mol.GetConformer()

    pos_lst = []
    symbol = []
    # Add atom indices
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        x, y = pos.x, pos.y
        pos_lst.append([x, y])
        symbol.append(atom.GetSymbol())

    x_min = min([x for x, y in pos_lst])
    x_max = max([x for x, y in pos_lst])
    y_min = min([y for x, y in pos_lst])
    y_max = max([y for x, y in pos_lst])

    scale_facor = 0.9 * size / max(x_max - x_min, y_max - y_min)

    for i, atom in enumerate(mol.GetAtoms()):
        # ax.text(x, y, str(atom.GetSymbol()), color="black", fontsize=12, ha='center', va='center')
        ax.text(pos_lst[i][0] * scale_facor + size / 2 + 10, size / 2 - pos_lst[i][1] * scale_facor - 10
                , str(atom.GetIdx()), color="red", fontsize=10, ha='center', va='center')

    plt.show()


def get_atoms_index(smiles: str, atom_type: str):
    """ given smiles and a atom type, return and index of all atom_type in the smiles"""

    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]

    # get atomic number of atom_type
    atom_num = Chem.GetPeriodicTable().GetAtomicNumber(atom_type)

    # get index of atom_type
    atom_index = [i for i, x in enumerate(atomic_numbers) if x == atom_num]

    return atom_index


def xtb_opt(structure: Atoms, fmax: float = 0.05, max_step: int = 200, fix_idx: list = None) -> Atoms:
    from ase.constraints import FixAtoms
    from ase.calculators.xtb import XTB
    from ase.optimize import BFGS
    # Define which atoms to fix. For example, fixing the first and second atoms:
    if fix_idx is not None:
        constraint = FixAtoms(indices=fix_idx)
        structure.set_constraint(constraint)

    # Setting up the calculator
    calculator = XTB(method="GFN2-xTB")  # Choose your method
    structure.set_calculator(calculator)

    # Optimization
    optimizer = BFGS(structure)
    optimizer.run(fmax=fmax, steps=max_step)

    return structure


def check_atoms_distance(structure: Atoms) -> float:
    # check if any two atoms in the structure is too close
    pos = structure.get_positions()

    # calculate distance matrix
    dst = np.linalg.norm(pos[:, np.newaxis, :] - pos[np.newaxis, :, :], axis=-1)

    min_dst = np.min(dst[np.triu_indices(len(dst), k=1)])

    return min_dst

