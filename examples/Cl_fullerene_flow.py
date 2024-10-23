# get fullerene Cl derivatives based on Cage
# 1. from Cage to get different addons (4)
# 2. addons to fullerene on the addon site
# 3. sort and check
# 4. get the lowest energy structures
# 5. add new addons on lowest energy structures, 1 by 1 or 2 by 2
# 6. loop until no more addons (40)
# 7. output all structures with addons and energies

from typing import List

import ase
import networkx as nx
import numpy as np
from dflow import (
    InputParameter,
    Inputs,
    OutputArtifact,
    OutputParameter,
    Outputs,
    Step,
    Steps,
    Workflow,
    argo_len,
    argo_sequence,
    set_config,
)
from dflow.python import PythonOPTemplate, Slices, upload_packages

import fullerenetool
from fullerenetool.fullerene.cage import FullereneCage
from fullerenetool.fullerene.derivatives import DerivativeGroup
from fullerenetool.operator.flow.ops.fullerene.nonisomorphic_addon import (
    establish_parrel_generate_nonisomorphic_addon_steps,
)
from fullerenetool.operator.flow.ops.fullerene.utils import GatherEnergies, IsomerSort
from fullerenetool.operator.flow.ops.model.mace import CalculateEnergy

# if "__file__" in locals():
#     upload_packages.append(__file__)
upload_packages.append(fullerenetool.__path__[0])
set_config(util_image="python:latest", save_path_as_parameter=True)

simple_machine_template_config = {
    "annotations": {
        "k8s.aliyun.com/eci-use-specs": "2-2Gi",  # 指定vCPU和内存，仅支持2 vCPU及以上规格
        "k8s.aliyun.com/eci-spot-strategy": "SpotWithPriceLimit",  # 采用系统自动出价，跟随当前市场实际价格
        "k8s.aliyun.com/eci-spot-price-limit": "0.1",  # 设置最高出价
        "k8s.aliyun.com/eci-spot-duration": "0",  # 设置无保护期
        "k8s.aliyun.com/eci-spot-release-strategy": "api-evict",  # 释放策略
    }
}
t4_gpu_machine_template_config = {
    "annotations": {
        "k8s.aliyun.com/eci-use-specs": "ecs.gn6i-c4g1.xlarge",
        # 指定vCPU和内存，仅支持2 vCPU及以上规格
        "k8s.aliyun.com/eci-spot-strategy": "SpotWithPriceLimit",  # 采用系统自动出价，跟随当前市场实际价格
        "k8s.aliyun.com/eci-spot-price-limit": "3.0",  # 设置最高出价
        "k8s.aliyun.com/eci-spot-duration": "0",  # 设置无保护期
        "k8s.aliyun.com/eci-spot-release-strategy": "api-evict",  # 释放策略
    }
}
gpu_machine_template_config = {
    "annotations": {
        "k8s.aliyun.com/eci-use-specs": "ecs.gn7i-c8g1.2xlarge",
        # 指定vCPU和内存，仅支持2 vCPU及以上规格
        "k8s.aliyun.com/eci-spot-strategy": "SpotWithPriceLimit",  # 采用系统自动出价，跟随当前市场实际价格
        "k8s.aliyun.com/eci-spot-price-limit": "3.0",  # 设置最高出价
        "k8s.aliyun.com/eci-spot-duration": "0",  # 设置无保护期
        "k8s.aliyun.com/eci-spot-release-strategy": "api-evict",  # 释放策略
    }
}
IMAGE = "registry-vpc.cn-heyuan.aliyuncs.com/xjtu-icp/fullerenetool:2024-10-24-00-38-18"


def add_exo_steps(
    simple_machine_template_config=simple_machine_template_config,
    gpu_machine_template_config=gpu_machine_template_config,
    group_size=5,
    image: str = IMAGE,
):
    add_steps = Steps(
        name="add-exo-steps",
        inputs=Inputs(
            parameters={
                "fulleren_init": InputParameter(),
                "addon": InputParameter(),
                "addon_start": InputParameter(),
                "start_idx_list": InputParameter(),
                "add_num": InputParameter(),
            }
        ),
        outputs=Outputs(
            parameters={
                "addons_index_list": OutputParameter(),
                "energy_list": OutputParameter(),
            },
            artifacts={
                "calculated_atoms_xyz": OutputArtifact(),
            },
        ),
    )
    candidategraph_list_step = Step(
        name="candidategraph-step",
        template=establish_parrel_generate_nonisomorphic_addon_steps(
            simple_machine_template_config=simple_machine_template_config,
            gpu_machine_template_config=gpu_machine_template_config,
            group_size=group_size,
            image=image,
            generate_addons_group_size=1,
        ),
        parameters={
            "fulleren_init": add_steps.inputs.parameters["fulleren_init"],
            "addon": add_steps.inputs.parameters["addon"],
            "addon_start": add_steps.inputs.parameters["addon_start"],
            "start_idx_list": add_steps.inputs.parameters["start_idx_list"],
            "add_num": add_steps.inputs.parameters["add_num"],
        },
        key="candidategraph-step",
    )
    add_steps.add(candidategraph_list_step)
    energy_step = Step(
        name="energy-step",
        template=PythonOPTemplate(
            CalculateEnergy,
            image=image,
            slices=Slices(
                input_parameter=[
                    "output_name",
                ],
                input_artifact=[
                    "atoms_file",
                ],
                output_parameter=[
                    "energy",
                    "time",
                ],
                output_artifact=["atoms_xyz"],
                group_size=group_size,
                pool_size=1,
            ),
            **gpu_machine_template_config,
        ),
        parameters={
            "output_name": candidategraph_list_step.outputs.parameters[
                "candidategraph_name_list"
            ],
            "optimize": True,
        },
        artifacts={
            "atoms_file": candidategraph_list_step.outputs.artifacts["atoms_file_list"]
        },
        key="energy-step",
        with_sequence=argo_sequence(
            argo_len(
                candidategraph_list_step.outputs.parameters["candidategraph_list"]
            ),
        ),
    )

    add_steps.add(energy_step)
    gather_energies = Step(
        name="gather-energies",
        template=PythonOPTemplate(
            GatherEnergies,
            image=image,
            **simple_machine_template_config,
        ),
        parameters={
            "candidategraph_list": candidategraph_list_step.outputs.parameters[
                "candidategraph_list"
            ],
        },
        artifacts={
            "calculated_atoms_xyz": energy_step.outputs.artifacts["atoms_xyz"],
        },
        key="gather-energies",
    )
    add_steps.add(gather_energies)
    sort = Step(
        name="sort",
        template=PythonOPTemplate(
            IsomerSort,
            image=image,
            **simple_machine_template_config,
        ),
        parameters={
            "addons_index_list": gather_energies.outputs.parameters[
                "addons_index_list"
            ],
            "energy_list": gather_energies.outputs.parameters["energy_list"],
            "pick_first_n": 0,
        },
        artifacts={
            "calculated_atoms_xyz": gather_energies.outputs.artifacts[
                "calculated_atoms_xyz"
            ],
        },
    )
    add_steps.add(sort)

    add_steps.outputs.parameters["addons_index_list"].value_from_parameter = (
        sort.outputs.parameters["addons_index_list"]
    )

    add_steps.outputs.parameters["energy_list"].value_from_parameter = (
        sort.outputs.parameters["energy_list"]
    )
    add_steps.outputs.artifacts["calculated_atoms_xyz"] = OutputArtifact(
        _from=sort.outputs.artifacts["calculated_atoms_xyz"]
    )

    return add_steps


def run(
    fulleren_init: FullereneCage,
    addon: DerivativeGroup,
    addon_start: int = 0,
    start_idx_list: List[List[int]] = [[]],
    addon_max: int = 40,
    addon_step: int = 1,
    addon_bond_length: float = 1.4,
    group_size=5,
    gpu_machine_template_config=gpu_machine_template_config,
    simple_machine_template_config=simple_machine_template_config,
):
    """
    Args:
        fulleren_init: FullereneCage
        addon_start: start addon number
        addon_max: max addon number
        addon_step: addon step
        addon_bond_length: bond length of addon
    """
    wf = Workflow(
        name="fullerene-dev-" + "cl",
    )
    addon_step = Step(
        name="addon-step",
        template=add_exo_steps(
            simple_machine_template_config=simple_machine_template_config,
            gpu_machine_template_config=gpu_machine_template_config,
            group_size=group_size,
            image=IMAGE,
        ),
        parameters={
            "fulleren_init": fulleren_init,
            "addon": addon,
            "addon_start": addon_start,
            "start_idx_list": start_idx_list,
            "add_num": addon_step,
        },
    )
    wf.add(addon_step)

    wf.submit()


if __name__ == "__main__":
    from ase.build import molecule

    C60 = molecule("C60")
    XCl = DerivativeGroup(
        atoms=ase.Atoms(
            "XF",
            [
                [0.1, 0.1, -0.2],
                [0, 0, 0],
                # [0, 0, 1.4]
            ],
        ),
        graph=nx.from_numpy_array(
            np.array(
                [
                    [
                        0,
                        1,
                    ],
                    [
                        1,
                        0,
                    ],
                ]
            )
        ),
        addon_atom_idx=0,
    )
    run(
        FullereneCage(C60),
        XCl,
        addon_start=2,
        start_idx_list=[[0, 8]],
        group_size=30,
        # addon_max=40,
        addon_step=1,
        # addon_bond_length=1.4
        gpu_machine_template_config=gpu_machine_template_config,
    )
