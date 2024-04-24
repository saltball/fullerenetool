import ase
import numpy as np
from ase import Atoms
from ase.build import molecule

from fullerenetool.logger import logger
from fullerenetool.operator.fullerene.add_exo import add_out_of_cage, get_cage_addon_pos


def test_add_exo():
    atoms = molecule("C60")
    fullerene = ase.Atoms(list(atoms))
    dev = fullerene
    for cage_idx in range(0, 60, 1):
        pos = np.array(
            [
                [0, 0, 0],
                [0.3566728, 0.5043982, 0.8736515],
                [0.3566728, 0.5043982, -0.8736515],
                [-1.07, 0, 0],
            ]
        )  # CH3
        dev_pos, dev_vec = get_cage_addon_pos(
            fullerene,
            cage_idx,
            Atoms("CH3X", [*pos, pos[0] + pos[1:].mean(axis=0)]),
            0,
            None,
        )

        dev = add_out_of_cage(
            dev,
            Atoms("CH3X", [*pos, pos[0] + pos[1:].mean(axis=0)]),
            addons_pos=dev_pos,
            addons_index=0,
            addons_vec=dev_vec,
            addons_conn_index=4,
            shake_num=50,
            check=True,
        )
        logger.info(dev)
