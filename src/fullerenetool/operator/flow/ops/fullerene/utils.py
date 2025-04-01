from pathlib import Path
from typing import List, Union

from dflow.python import OP, OPIO, Artifact, BigParameter, OPIOSign


class GatherEnergies(OP):
    """
    Gather the energies of the addons.
    """

    def __init__(self):
        pass

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "candidategraph_list": List[str],
                "calculated_atoms_xyz": Artifact(List[Path]),
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
        from ase.io.extxyz import read_extxyz

        candidategraph_list = op_in["candidategraph_list"]
        calculated_atoms_xyz = op_in["calculated_atoms_xyz"]

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
    pick_first_n : float or int
        The number of addons to be picked, set to 0 to pick all.
        Negative float number means using the energy less than pick_first_n.
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
                "addon_pos_index_list": BigParameter(List[List[int]]),
                "pick_first_n": Union[int, float],
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "addons_index_list": BigParameter(List[str]),
                "energy_list": BigParameter(List[float]),
                "calculated_atoms_xyz": Artifact(List[Path]),
                "addon_pos_index_list": List[List[int]],
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
        addon_pos_index_list = op_in["addon_pos_index_list"]
        pick_first_n = op_in["pick_first_n"]
        if pick_first_n > 0:
            pick_first_n = int(pick_first_n)
            sorted_data = sorted(
                zip(
                    addons_index_list,
                    energy_list,
                    calculated_atoms_xyz,
                    addon_pos_index_list,
                ),
                key=lambda x: x[1],
                reverse=False,
            )[:pick_first_n]
        elif pick_first_n < 0:
            threshold = abs(pick_first_n)
            min_energy = min(energy_list)
            filtered_data = [
                (idx, energy, atoms, pos_idx)
                for idx, energy, atoms, pos_idx in zip(
                    addons_index_list,
                    energy_list,
                    calculated_atoms_xyz,
                    addon_pos_index_list,
                )
                if energy - min_energy < threshold
            ]
            sorted_data = sorted(
                filtered_data,
                key=lambda x: x[1],
                reverse=False,
            )

        else:
            raise RuntimeError(f"pick_first_n={pick_first_n} is invalid.")

        return OPIO(
            {
                "addons_index_list": [x[0] for x in sorted_data],
                "energy_list": [x[1] for x in sorted_data],
                "calculated_atoms_xyz": [Path(x[2]) for x in sorted_data],
                "addon_pos_index_list": [x[3] for x in sorted_data],
            }
        )
