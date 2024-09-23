import torch
import numpy as np
import itertools

def get_rotations(so3_discretization_type='icosahedron'):
    """
    Generate rotation matrices discretizing the SO(3) group based on the chosen Platonic solid.

    Parameters:
    - so3_discretization_type (str): Type of solid to base the discretization on.
      Options are 'tetrahedron', 'cube', 'octahedron', 'icosahedron', 'dodecahedron'.

    Returns:
    - rotations (torch.Tensor): A tensor of shape (N, 3, 3), where N is the number of rotations,
      containing rotation matrices.

    The order of discretization (number of rotations) for each solid:
    - Tetrahedron: 12 rotations
    - Cube/Octahedron: 24 rotations
    - Icosahedron/Dodecahedron: 60 rotations
    """
    rotations = []
    
    if so3_discretization_type == 'tetrahedron':
        # Order of discretization: 12 rotations
        # Generate the 12 rotation matrices of the tetrahedral group
        # Identity rotation
        rotations.append(np.eye(3))

        # 180-degree rotations about the axes
        rotations.extend([
            np.diag([1, -1, -1]),  # 180° about X-axis
            np.diag([-1, 1, -1]),  # 180° about Y-axis
            np.diag([-1, -1, 1])   # 180° about Z-axis
        ])

        # 120-degree rotations around body diagonals
        angle = 2 * np.pi / 3  # 120 degrees
        axes = np.array([
            [1, 1, 1],
            [1, -1, -1],
            [-1, 1, -1],
            [-1, -1, 1]
        ])
        axes = axes / np.linalg.norm(axes, axis=1, keepdims=True)

        for axis in axes:
            # 120-degree rotation
            R = rotation_matrix(axis, angle)
            rotations.append(R)
            # 240-degree rotation (inverse rotation)
            R_inv = rotation_matrix(axis, -angle)
            rotations.append(R_inv)

        rotations = np.stack(rotations)

    elif so3_discretization_type in ['cube', 'octahedron']:
        # Order of discretization: 24 rotations
        # Generate all 24 rotation matrices of the octahedral group
        # Generate all permutations of the axes with possible sign flips
        perms = list(itertools.permutations([0, 1, 2]))
        signs = [-1, 1]
        for perm in perms:
            for sign_combo in itertools.product(signs, repeat=3):
                R = np.zeros((3, 3))
                for i in range(3):
                    R[i, perm[i]] = sign_combo[i]
                if np.linalg.det(R) > 0.5:  # Ensure rotation matrix (determinant +1)
                    rotations.append(R)

        rotations = np.stack(rotations)

    elif so3_discretization_type in ['icosahedron', 'dodecahedron']:
        # Golden ratio
        phi = (1 + np.sqrt(5)) / 2

        # Possible values for quaternion components
        components = [0, 1, phi]
        signs = [-1, 1]

        quaternions = set()

        # Generate all unique quaternions corresponding to the icosahedral group rotations
        for perm in set(itertools.permutations(components)):
            for sign_combo in itertools.product(signs, repeat=3):
                q = np.array([0, sign_combo[0]*perm[0], sign_combo[1]*perm[1], sign_combo[2]*perm[2]])
                if np.linalg.norm(q[1:]) == 0:
                    continue
                q /= np.linalg.norm(q)
                quaternions.add(tuple(np.round(q, decimals=5)))

        quaternions = np.array(list(quaternions))

        # Convert quaternions to rotation matrices
        rotations = []
        for q in quaternions:
            R = quaternion_to_rotation_matrix(q)
            rotations.append(R)

        # Remove duplicates and ensure we have 60 rotations
        rotations = np.array(rotations)
        rotations = np.unique(np.round(rotations, decimals=5), axis=0)

        if rotations.shape[0] != 60:
            raise ValueError(f"Expected 60 rotations, but got {rotations.shape[0]}")

    else:
        raise ValueError(f"Unsupported SO3 discretization type: {so3_discretization_type}")

    rotations = torch.from_numpy(rotations).float()
    return rotations

def rotation_matrix(axis, angle):
    """
    Compute the rotation matrix for a given axis and angle using Rodrigues' formula.

    Parameters:
    - axis (ndarray): A 3-element array representing the rotation axis (should be normalized).
    - angle (float): The rotation angle in radians.

    Returns:
    - R (ndarray): A 3x3 rotation matrix.
    """
    axis = axis / np.linalg.norm(axis)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

def quaternion_to_rotation_matrix(q):
    """
    Convert a quaternion into a rotation matrix.

    Parameters:
    - q (ndarray): Quaternion in the form [q0, q1, q2, q3]

    Returns:
    - R (ndarray): 3x3 rotation matrix
    """
    q0, q1, q2, q3 = q
    R = np.array([
        [1 - 2*(q2**2 + q3**2),     2*(q1*q2 - q0*q3),     2*(q1*q3 + q0*q2)],
        [    2*(q1*q2 + q0*q3), 1 - 2*(q1**2 + q3**2),     2*(q2*q3 - q0*q1)],
        [    2*(q1*q3 - q0*q2),     2*(q2*q3 + q0*q1), 1 - 2*(q1**2 + q2**2)]
    ])
    return R
