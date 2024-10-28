from pathlib import Path
from typing import List

from dflow.python import OP, OPIO, Artifact, BigParameter, OPIOSign


@OP.function
def GatherEnergies(
    candidategraph_list: List[str],
    calculated_atoms_xyz: Artifact(List[Path]),
) -> {
    "addons_index_list": BigParameter(List[str]),
    "energy_list": BigParameter(List[float]),
    "calculated_atoms_xyz": Artifact(List[Path]),
}:
    """
    Gather the energies of the addons.
    """
    from ase.io.extxyz import read_extxyz

    addons_index_list = []
    energy_list = []
    for i, candidategraph in enumerate(candidategraph_list):
        addons_index_list.append(candidategraph)
        atoms = list(read_extxyz(calculated_atoms_xyz[i].open("r")))[-1]
        energy_list.append(float(atoms.get_potential_energy()))
    return {
        "addons_index_list": addons_index_list,
        "energy_list": energy_list,
        "calculated_atoms_xyz": calculated_atoms_xyz,
    }


class IsomerSort(OP):
    """
    Sort the addons by energy.

    Parameters
    pick_first_n : int
        The number of addons to be picked, set to 0 to pick all.
        Negative number means 0.
    """

    def __init__(self):
        pass

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "addons_index_list": BigParameter(List[str]),
                "energy_list": BigParameter(List[float]),
                "calculated_atoms_xyz": Artifact(List[Path]),
                "pick_first_n": int,
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "addons_index_list": BigParameter(List[str]),
                "energy_list": BigParameter(List[float]),
                "calculated_atoms_xyz": Artifact(List[Path]),
            }
        )

    @OP.exec_sign_check
    def execute(
        self,
        op_in: OPIO,
    ) -> OPIO:
        addons_index_list = op_in["addons_index_list"]
        energy_list = op_in["energy_list"]
        calculated_atoms_xyz = op_in["calculated_atoms_xyz"]
        pick_first_n = op_in["pick_first_n"]
        if pick_first_n > 0:
            sorted_data = sorted(
                zip(addons_index_list, energy_list, calculated_atoms_xyz),
                key=lambda x: x[1],
                reverse=False,
            )[:pick_first_n]
        else:
            sorted_data = sorted(
                zip(addons_index_list, energy_list, calculated_atoms_xyz),
                key=lambda x: x[1],
                reverse=False,
            )

        return OPIO(
            {
                "addons_index_list": [x[0] for x in sorted_data],
                "energy_list": [x[1] for x in sorted_data],
                "calculated_atoms_xyz": [Path(x[2]) for x in sorted_data],
            }
        )
