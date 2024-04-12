from pathlib import Path
from typing import Optional

import ase
import numpy as np
from ase.data import covalent_radii

from fullerenetool.fullerene import BaseFullerene, FullereneCage
from fullerenetool.fullerene.derivatives import FullereneDerivative
from fullerenetool.logger import setup_logger
from fullerenetool.operator.checker.molecule import valid_molecule

logger = setup_logger(
    Path(__file__).name,
)


def addoutofcage(
    cage: BaseFullerene,
    addons: ase.Atoms,
    addons_pos: Optional[np.ndarray] = None,
    addons_index: int = 0,
) -> FullereneDerivative:
    """Add exogenous functionalization to the fullerene cage.

    Args:
        cage (BaseFullerene): The base fullerene cage.
        addons (ase.Atoms): The atoms representing the exogenous functionalization.
        addons_pos (np.ndarray, optional): The positions of the exogenous
            functionalization atoms. Related to the `cage`.
            Defaults to None means use the addons as input without transformation.
        addons_index (int, optional): The index of the exogenous functionalization.
            Defaults to None means the first atom in the exogenous functionalization
            `addons` will be added to fullerene.

    Returns:
        FullereneDerivative: The fullerene derivative with the exogenous
            functionalization added.
    """
    valid_molecule(addons)
    logger.info(
        "[addoncage]Try to add {} on cage {}".format(
            ",".join(
                ["{}({})".format(atom.symbol, list(atom.position)) for atom in addons]
            ),
            cage.atoms.symbols,
        )
    )
    if addons_pos is not None:  # move the addons_pos
        logger.debug("[addoncage] addons_pos is not None, use ({})".format(addons_pos))
        move_vector = addons_pos - addons.positions[addons_index]
        addons.positions += move_vector
    logger.debug(
        "[addoncage] use addons [{}]".format(
            ", ".join(
                [
                    "{}({})".format(
                        atom.symbol,
                        ",".join("{:.2f}".format(pos) for pos in atom.position),
                    )
                    for atom in addons
                ]
            )
        )
    )
    atoms = ase.Atoms(cage.atoms + addons)
    valid_molecule(atoms)
    return FullereneDerivative(atoms)


def get_cage_addon_pos(
    cage: FullereneCage,
    index: int,
    addons: ase.Atoms,
    addon_index: Optional[int] = None,
    bond_length=None,
) -> np.ndarray:
    """Get the position of the exogenous functionalization on the fullerene cage."""
    neighbors = np.argsort(
        cage.atoms.get_distances(index, None),
    )[:4]
    sphere_center = sphere_center_of_four_points(*cage.atoms.positions[neighbors])
    addon_direct = (cage.atoms.positions[index] - sphere_center) / np.linalg.norm(
        cage.atoms.positions[index] - sphere_center
    )
    if bond_length is not None:
        pass
    else:
        bond_length = (
            covalent_radii[cage.atoms.get_atomic_numbers()[index]]
            + covalent_radii[
                addons.get_atomic_numbers()[
                    addon_index if addon_index is not None else 0
                ]
            ]
        )
    return cage.atoms.positions[index] + addon_direct * bond_length


def sphere_center_of_four_points(
    point_a: np.ndarray, point_b: np.ndarray, point_c: np.ndarray, point_d: np.ndarray
) -> np.ndarray:
    r"""Calculate the center of sphere for four non-coplanar points

    Parameters
    ----------
    point_a: np.ndarray
        point coordinate
    point_b: np.ndarray
        point coordinate
    point_c: np.ndarray
        point coordinate
    point_d: np.ndarray
        point coordinate

    Returns
    -------
    np.ndarray
        coordinate of sphere center

    Notes
    -----
    This function based on these equations:

    .. math:: \mathbf{OD}*2(\mathbf{DA}+\mathbf{DB}+\mathbf{DC})=-(|DA|^2+|DB|^2+|DC|^2)


    Same for point :math:`A,B,C`, which leads us to a matrix form

    .. math:: \mathbf{Or'}\mathbf{O}^T=[A*O'_A+|AD|^2+|AB|^2+|AC|^2,
        B*O'_B+|BA|^2+|BC|^2+|BD|^2,C*O'_C+|CA|^2+|CB|^2+|CD|^2]^T

    where :math:`O` is the centor of the sphere, :math:`\mathbf{O'}` is the
        matrix as below:

    .. math:: \mathbf{O'}=[\mathbf{AD+AB+AC},\mathbf{BA+BC+BD},\mathbf{CA+CB+CD}]^T
    """
    pos_list = np.array([point_a, point_b, point_c, point_d])
    pos_tensor = pos_list - pos_list[:, None, :]
    dis_matrix = np.linalg.norm(pos_tensor, axis=-1)
    center_matrix = pos_tensor.sum(axis=-2)
    B = (pos_list * center_matrix).sum(axis=-1) + ((dis_matrix**2).sum(axis=-1)) / 2
    sphere_center = np.linalg.inv(center_matrix[:3]) @ B[:3]
    return sphere_center
