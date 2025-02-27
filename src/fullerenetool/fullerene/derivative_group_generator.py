from enum import Enum

import numpy as np


class DerivativeGroupType(Enum):
    """
    Enum class `DerivativeGroupType` defines preset substituent geometries
    with standardized coordinate systems. Each geometry type includes a template
    of atoms and their base coordinates following these conventions:
    - The substitution point (X) is at coordinates (0, 0, -1).
    - The anchor atom (bonded to X) is at the origin (0, 0, 0).
    - Subsequent atoms are positioned along the +Z axis by default.

    Attributes:
        SINGLE (dict):
            -X-A
        LINEAR (dict):
            -X-A-H
        BENT (dict):
            -X-A╮
                H
        TRIGONAL_PYRAMIDAL (dict):
                H
            -X-A╉(electron pair)
                H
        TRIGONAL_PLANAR (dict):
                H
            -X-A┫
                H
        TETRAHEDRAL (dict):
                H
            -X-A╋H
                H
    """

    SINGLE = {
        "template": ["X", "C"],
        "base_coords": np.array(
            [
                [0, 0, -1],  # X position
                [0, 0, 0],  # Anchor atom
            ]
        ),
        "topo": [
            [0, 1],
            [1, 0],
        ],
    }
    LINEAR = {
        "template": ["X", "C", "N"],
        "base_coords": np.array(
            [
                [0, 0, -1],  # X position
                [0, 0, 0],  # Anchor atom
                [0, 0, 1],  # Terminal atom
            ]
        ),
        "topo": [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
    }
    BENT = {
        "template": ["X", "O", "H"],
        "base_coords": np.array([[0, 0, -1], [0, 0, 0], [0, 1, 0]]),
        "topo": [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
    }
    TRIGONAL_PYRAMIDAL = {
        "template": ["X", "N", "H", "H"],
        "base_coords": np.array(
            [[0, 0, -1], [0, 0, 0], [0.94, 0.4, 0.3], [-0.94, 0.4, 0.3]]
        ),
        "topo": [
            [0, 1, 0, 0],
            [1, 0, 1, 1],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
        ],
    }
    TRIGONAL_PLANAR = {
        "template": ["X", "B", "H", "H"],
        "base_coords": np.array(
            [[0, 0, -1], [0, 0, 0], [0.9, 0, 0.52], [-0.9, 0, 0.52]]
        ),
        "topo": [
            [0, 1, 0, 0],
            [1, 0, 1, 1],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
        ],
    }
    TETRAHEDRAL = {
        "template": ["X", "C", "H", "H", "H"],
        "base_coords": np.array(
            [
                [0, 0, -1],
                [0, 0, 0],
                [0.94, 0, 0.34],
                [-0.47, 0.82, 0.34],
                [-0.47, -0.82, 0.34],
            ]
        ),
        "topo": [
            [
                0,
                1,
                0,
                0,
                0,
            ],
            [
                1,
                0,
                1,
                1,
                1,
            ],
            [
                0,
                1,
                0,
                0,
                0,
            ],
            [
                0,
                1,
                0,
                0,
                0,
            ],
            [
                0,
                1,
                0,
                0,
                0,
            ],
        ],
    }


def _apply_random_rotation(
    coords: np.ndarray, anchor_idx: int = 1, max_angle: float = 30
) -> np.ndarray:
    """
    Apply random rotation to substituent atoms while keeping X-anchor vector fixed

    Args:
        coords: Input coordinates (will be modified in-place)
        anchor_idx: Index of anchor atom (must remain at origin)
        max_angle: Maximum rotation angle in degrees

    Returns:
        Rotated coordinates array
    """
    # 生成随机旋转轴（垂直于Z轴）
    theta = np.radians(np.random.uniform(-max_angle, max_angle))
    axis = np.random.choice(["x", "y"])  # 随机选择X或Y轴为旋转轴

    # 创建旋转矩阵（仅绕选定轴旋转）
    c, s = np.cos(theta), np.sin(theta)
    if axis == "x":
        rot_matrix = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    else:
        rot_matrix = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    # 仅旋转取代基原子（锚点之后的原子）
    substituent_coords = coords[anchor_idx + 1 :] - coords[anchor_idx]
    rotated = np.dot(substituent_coords, rot_matrix.T)

    # 重组坐标并返回
    coords[anchor_idx + 1 :] = rotated + coords[anchor_idx]
    return coords


def generate_substituent_coords(
    substituent: str,
    structure_type: DerivativeGroupType,
    random_orient: bool = False,
    max_rotation_angle: float = 30.0,
) -> np.ndarray:
    """
    Generate substituent coordinates with optional random orientation

    Args:
        substituent: Chemical formula starting with X (e.g. "XCH3")
        structure_type: Geometry template from StructureType
        random_orient: Enable random orientation (default: False)
        max_rotation_angle: Maximum rotation angle in degrees (default: 30)

    Returns:
        Array of shape (N,3) with coordinates ordered as [X, anchor, *others]

    Raises:
        ValueError: If substituent pattern doesn't match template
    """
    # 验证输入格式
    elements = [c for c in substituent if c.isupper()]
    if elements[0] != "X":
        raise ValueError("Substituent must start with X")

    base_coords = structure_type.value["base_coords"].copy()

    if random_orient:
        base_coords = _apply_random_rotation(base_coords, max_angle=max_rotation_angle)

    return base_coords
