import random
from pathlib import Path
from typing import Optional, Set, Union

import ase
import numpy as np
from ase.data import covalent_radii
from scipy.spatial.transform import Rotation
from tenacity import retry, retry_if_exception_type, stop_after_attempt

from fullerenetool.logger import setup_logger
from fullerenetool.operator.checker.molecule import (
    AtomTooCloseError,
    filter_ghost_atom,
    valid_molecule,
)

logger = setup_logger(
    Path(__file__).name,
)


def add_out_of_cage(
    atoms: ase.Atoms,
    addons: ase.Atoms,
    *,
    addons_pos: Optional[np.ndarray] = None,
    addons_vec: Optional[np.ndarray] = None,
    addons_index: int = 0,
    addons_conn_index: Optional[int] = None,
    shake_num: int = 50,
    check=True,
) -> ase.Atoms:
    """Add exogenous functionalization to the fullerene cage.

    This function takes a base fullerene cage and adds exogenous functionalization
    to it. The exogenous functionalization is represented by a separate set of
    atoms (`addons`). The positions and vectors of the exogenous functionalization
    atoms can be optionally provided to control their placement and rotation.

    Args:
        atoms (ase.Atoms): The base fullerene cage.
        addons (ase.Atoms): The atoms representing the exogenous functionalization.
        addons_pos (np.ndarray, optional): The positions of the exogenous
            functionalization atoms. Related to the `atoms`. Defaults to None,
            which means use the `addons` as input without transformation.
        addons_vec (np.ndarray, optional): The vectors representing the exogenous
            functionalization atoms. It means the vector from `addons_index` atom to
            the first connect atom (if exist) for the addons to rotation.
            Defaults to None, which means use the `addons` as input without rotation.
        addons_index (int, optional): The index of the exogenous functionalization.
            Defaults to 0, which means the first atom in the exogenous functionalization
            `addons` will be added to the fullerene.
        addons_conn_index (int, optional): The index of the first connected atom for
            the exogenous functionalization. If not provided, the first connected atom
            will be determined automatically based on the connectivity of the `addons`.
        shake_num (int): Whether to shake the atoms after adding the exogenous
            functionalization. Try randomly rotate the addon aroung the addons_vec until
            check pass with up to `shake_num` times. Defaults to 50.
        check (bool, optional): Whether to check the validity of the resulting molecule
            after adding the exogenous functionalization. Defaults to True.

    Returns:
        ase.Atoms: The fullerene with the exogenous functionalization added.
    """
    if check:
        valid_molecule(addons)
    logger.debug(
        "[addoncage] Try to add {} on cage {}".format(
            ",".join(
                ["{}({})".format(atom.symbol, list(atom.position)) for atom in addons]
            ),
            atoms.symbols,
        )
        + " with addons_pos={}, addons_vec={},"
        " addons_index={}, addons_conn_index={}".format(
            addons_pos, addons_vec, addons_index, addons_conn_index
        )
    )
    # only rotate when addons more than one atom
    if len(addons) > 1:
        # rotate the addons[addons_index, first_conn] to addons_vec
        if addons_vec is not None:
            logger.debug(
                "[addoncage] addons_vec is not None, use ({})".format(addons_vec)
            )
            if addons_conn_index is None:
                addons_conn_index = int(
                    np.argsort(
                        np.linalg.norm(
                            addons.positions - addons.positions[addons_index], axis=1
                        )
                    )[1]
                )
            else:
                logger.debug(
                    "[addoncage] addons_conn_index is not None, use [{}]{}".format(
                        addons[addons_conn_index].symbol, addons_conn_index
                    )
                )
            logger.debug(
                "[addoncage] addons_vec is used for rotate [{}]".format(
                    ", ".join(
                        [
                            "{}({})".format(
                                atom.symbol,
                                list(atom.position),
                            )
                            for index in [addons_index, addons_conn_index]
                            for atom in [addons[index]]
                        ]
                    )
                )
                + " in addons"
            )
            # rotate the addons[addons_index, first_conn] to addons_vec
            # get the rotation matrix

            target_atoms_vec = (
                addons.positions[addons_conn_index] - addons.positions[addons_index]
            )
            target_atoms_vec = target_atoms_vec / np.linalg.norm(target_atoms_vec)
            addons_vec = addons_vec / np.linalg.norm(addons_vec)
            rotation_matrix, _ = Rotation.align_vectors(addons_vec, target_atoms_vec)
            logger.debug(
                "generate rotation from {} to {}".format(target_atoms_vec, addons_vec)
            )
            rotated_point = (rotation_matrix.as_matrix() @ addons.positions.T).T
            logger.debug("{} rorated to {}".format(addons.positions, rotated_point))
            addons.positions = rotated_point

            logger.debug(
                "addons_vec: {}".format(addons_vec / np.linalg.norm(addons_vec)),
            )
    else:
        if addons_vec is not None:
            logger.info("addons_vec of one atom addons is skipped.")

    # move the addons_pos
    if addons_pos is not None:
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

    try_atoms = _add_atom_and_check.retry_with(
        stop=stop_after_attempt(shake_num),
        before=lambda x: logger.debug("retry to add:{}".format(x)),
    )(atoms, addons, addons_vec, addons[addons_index].position)
    try_atoms = filter_ghost_atom(try_atoms)
    if check:
        valid_molecule(try_atoms)
    return try_atoms


@retry(
    stop=stop_after_attempt(50),
    retry=retry_if_exception_type(AtomTooCloseError),
)
def _add_atom_and_check(atoms, addons, vec, center_point) -> ase.Atoms:
    addons = _random_rotate_addons(addons, vec, center_point)
    try_atoms = ase.Atoms(atoms + addons)
    try_atoms = filter_ghost_atom(try_atoms)
    valid_molecule(try_atoms)
    return try_atoms


def _random_rotate_addons(
    addons: ase.Atoms,
    vec: np.ndarray,
    center_point: Optional[np.ndarray] = None,
):
    """Rotate the addons around vec and center_point randomly.

    Args:
        addons (ase.Atoms): The addons to be rotated.
        vec (np.ndarray): The vector around which the addons will be rotated.
        center_point (Optional[np.ndarray], optional): The center point for rotation.
            Defaults to None for use the index=0 atom in addons.

    Returns:
        try_addons: The rotated addons.

    """
    if center_point is None:
        center_point = addons[0].position
    tmpatoms = addons.copy()
    tmpatoms.rotate(random.randrange(1, 180, 1), vec, center_point)
    return tmpatoms


def get_cage_addon_pos(
    cage: ase.Atoms,
    index: int,
    addons: ase.Atoms,
    addon_index: Optional[int] = None,
    bond_length=None,
    return_vec=True,
) -> Union[np.ndarray, Set[np.ndarray]]:
    """Get the position of the exogenous functionalization on the fullerene cage.

    Args:
        cage (ase.Atoms): The fullerene cage.
        index (int): The index of the atom on the fullerene cage.
        addons (ase.Atoms): The exogenous functionalization atoms.
        addon_index (Optional[int], optional): The index of the exogenous
            functionalization atom. Defaults to None.
        bond_length ([type], optional): The bond length between the fullerene atom
            and the exogenous functionalization atom. Defaults to None which means
            using the covalent_radii of atoms
        return_vec (bool, optional): Whether to return the vector along with the
            position. Defaults to True.

    Returns:
        Union[np.ndarray, Set[np.ndarray]]: The position of the exogenous
            functionalization on the fullerene cage.

    """
    neighbors = np.argsort(
        cage.get_distances(index, None),
    )[:4]
    sphere_center = sphere_center_of_four_points(*cage.positions[neighbors])
    addon_direct = (cage.positions[index] - sphere_center) / np.linalg.norm(
        cage.positions[index] - sphere_center
    )
    if bond_length is not None:
        pass
    else:
        bond_length = (
            covalent_radii[cage.get_atomic_numbers()[index]]
            + covalent_radii[
                addons.get_atomic_numbers()[
                    addon_index if addon_index is not None else 0
                ]
            ]
        )
    if return_vec:
        return cage.positions[index] + addon_direct * bond_length, addon_direct
    else:
        return cage.positions[index] + addon_direct * bond_length


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
