# get fullerene Cl derivatives based on Cage
# 1. from Cage to get different addons (4)
# 2. addons to fullerene on the addon site
# 3. sort and check
# 4. get the lowest energy structures
# 5. add new addons on lowest energy structures, 1 by 1 or 2 by 2
# 6. loop until no more addons (40)
# 7. output all structures with addons and energies

from pathlib import Path
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
    upload_artifact,
)
from dflow.python import OP, PythonOPTemplate, Slices, upload_packages

import fullerenetool
from fullerenetool.fullerene.cage import FullereneCage
from fullerenetool.fullerene.derivatives import DerivativeGroup
from fullerenetool.operator.flow.ops.fullerene.nonisomorphic_addon import (
    establish_parrel_generate_nonisomorphic_addon_steps,
)
from fullerenetool.operator.flow.ops.fullerene.utils import GatherEnergies, IsomerSort
from fullerenetool.operator.flow.ops.model.dp import DPCalculateEnergy

# if "__file__" in locals():
#     upload_packages.append(__file__)
upload_packages.append(fullerenetool.__path__[0])
set_config(
    util_image="python:latest",
    # save_path_as_parameter=True
)

simple_machine_template_config = {
    "annotations": {
        "k8s.aliyun.com/eci-use-specs": "2-2Gi",  # 指定vCPU和内存，仅支持2 vCPU及以上规格
        "k8s.aliyun.com/eci-spot-strategy": "SpotWithPriceLimit",  # 采用系统自动出价，跟随当前市场实际价格
        "k8s.aliyun.com/eci-spot-price-limit": "0.1",  # 设置最高出价
        "k8s.aliyun.com/eci-spot-duration": "0",  # 设置无保护期
        "k8s.aliyun.com/eci-spot-release-strategy": "api-evict",  # 释放策略
        "k8s.aliyun.com/eci-fail-strategy": "fail-back",
    }
}
gpu_machine_template_config = {
    "annotations": {
        "k8s.aliyun.com/eci-use-specs": "ecs.gn6i-c4g1.xlarge,ecs.gn6e-c12g1.3xlarge,ecs.gn7i-c8g1.2xlarge",
        # 指定vCPU和内存，仅支持2 vCPU及以上规格
        "k8s.aliyun.com/eci-spot-strategy": "SpotWithPriceLimit",  # 采用系统自动出价，跟随当前市场实际价格
        "k8s.aliyun.com/eci-spot-price-limit": "2.5",  # 设置最高出价
        "k8s.aliyun.com/eci-spot-duration": "0",  # 设置无保护期
        "k8s.aliyun.com/eci-spot-release-strategy": "api-evict",  # 释放策略
    }
}
IMAGE = "registry-vpc.cn-heyuan.aliyuncs.com/xjtu-icp/fullerenetool:2024-10-24-00-38-18"
DP_IMAGE = "registry-vpc.cn-heyuan.aliyuncs.com/xjtu-icp/deepmd-kit:latest"


@OP.function
def schedule_next(
    addon_start: int,
    addon_max: int,
    add_num: int,
) -> {"addon_next": int, "finished": bool}:
    return {
        "addon_next": addon_start + add_num,
        "finished": addon_start + add_num >= addon_max,
    }


def add_exo_steps(
    addon_max,
    model_file,
    simple_machine_template_config=simple_machine_template_config,
    gpu_machine_template_config=gpu_machine_template_config,
    group_size=100,
    generate_addons_group_size=5,
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
                "pick_first_n": InputParameter(),
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
            generate_addons_group_size=generate_addons_group_size,
        ),
        parameters={
            "fulleren_init": add_steps.inputs.parameters["fulleren_init"],
            "addon": add_steps.inputs.parameters["addon"],
            "addon_start": add_steps.inputs.parameters["addon_start"],
            "start_idx_list": add_steps.inputs.parameters["start_idx_list"],
            "add_num": add_steps.inputs.parameters["add_num"],
        },
        key="candidategraph-step-%s" % add_steps.inputs.parameters["addon_start"],
    )
    add_steps.add(candidategraph_list_step)
    energy_step = Step(
        name="energy-step",
        template=PythonOPTemplate(
            DPCalculateEnergy,
            image=DP_IMAGE,
            slices=Slices(
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
            "optimize": True,
        },
        artifacts={
            "atoms_file": candidategraph_list_step.outputs.artifacts["atoms_file_list"],
            "model_file": upload_artifact(model_file),
        },
        key="energy-step-%s-%s"
        % (add_steps.inputs.parameters["addon_start"], "{{item}}"),
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
        key="gather-energies-%s" % add_steps.inputs.parameters["addon_start"],
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
            "addon_pos_index_list": candidategraph_list_step.outputs.parameters[
                "addon_pos_index_list"
            ],
            "pick_first_n": add_steps.inputs.parameters["pick_first_n"],
        },
        artifacts={
            "calculated_atoms_xyz": gather_energies.outputs.artifacts[
                "calculated_atoms_xyz"
            ],
        },
        key="sort-%s" % add_steps.inputs.parameters["addon_start"],
    )
    add_steps.add(sort)

    schedule_next_step = Step(
        name="schedule-next",
        template=PythonOPTemplate(schedule_next, image=image),
        parameters={
            "addon_start": add_steps.inputs.parameters["addon_start"],
            "addon_max": addon_max,
            "add_num": add_steps.inputs.parameters["add_num"],
        },
        # key="schedule-next-%s" % add_steps.inputs.parameters["addon_start"]
    )
    add_steps.add(schedule_next_step)

    next_addon_steps = Step(
        name="next-addon-steps",
        template=add_steps,
        parameters={
            "fulleren_init": add_steps.inputs.parameters["fulleren_init"],
            "addon": add_steps.inputs.parameters["addon"],
            "addon_start": schedule_next_step.outputs.parameters["addon_next"],
            "start_idx_list": sort.outputs.parameters["addon_pos_index_list"],
            "add_num": add_steps.inputs.parameters["add_num"],
            "pick_first_n": add_steps.inputs.parameters["pick_first_n"],
        },
        when="%s == false" % (schedule_next_step.outputs.parameters["finished"]),
    )
    add_steps.add(next_addon_steps)

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
    flow_name,
    fulleren_init: FullereneCage,
    addon: DerivativeGroup,
    model_file: Path,
    addon_start: int = 0,
    start_idx_list: List[List[int]] = [[]],
    addon_max: int = 40,
    addon_step: int = 1,
    addon_bond_length: float = 1.4,
    group_size=5,
    generate_addons_group_size=50,
    pick_first_n=0,
    gpu_machine_template_config=gpu_machine_template_config,
    simple_machine_template_config=simple_machine_template_config,
    reuse_step=None,
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
        name=flow_name,
    )
    addon_step = Step(
        name="addon-step",
        template=add_exo_steps(
            addon_max=addon_max,
            model_file=model_file,
            simple_machine_template_config=simple_machine_template_config,
            gpu_machine_template_config=gpu_machine_template_config,
            group_size=group_size,
            generate_addons_group_size=generate_addons_group_size,
            image=IMAGE,
        ),
        parameters={
            "fulleren_init": fulleren_init,
            "addon": addon,
            "addon_start": addon_start,
            "start_idx_list": start_idx_list,
            "add_num": addon_step,
            "pick_first_n": pick_first_n,
        },
        # key="addon-step-%s" % addon_start
    )
    wf.add(addon_step)

    wf.submit(reuse_step=reuse_step)


if __name__ == "__main__":
    import os
    from pathlib import Path
    from pprint import pprint

    import pynauty as pn
    from ase.build import molecule
    from ase.io.extxyz import read_extxyz

    os.chdir(Path(__file__).parent)

    from fullerenetool.operator.graph import nx_to_nauty

    # fullerene_atoms = molecule("C60")
    fullerene_atoms = list(
        read_extxyz(Path("fullerene_xyz/C70_000008149opt.xyz").open("r"))
    )[-1]
    addon_mol = DerivativeGroup(
        atoms=ase.Atoms(
            "XOH",
            [[0.5, 0.5, 0.0], [0, 0, 0], [0, 0, 1.4]],
        ),
        graph=nx.from_numpy_array(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])),
        addon_atom_idx=0,
    )
    fulleren_init = FullereneCage(fullerene_atoms)

    canon_index = pn.canon_label(
        nx_to_nauty(fulleren_init.graph.graph, include_z_labels=False)
    )

    fulleren_init = FullereneCage(fullerene_atoms[np.array(canon_index)])

    reuse_step_list = []

    # steps = Workflow(id="fullerene-dev-cage-c60-f-1bx13").query_step()
    # for step in steps:
    #     if step["key"] is not None:
    #         reuse_step_list.append(step)
    max_addon_num = 20
    run(
        "fullerene-c70-dev-max{}-{}".format(
            max_addon_num,
            (fulleren_init.name + "-" + addon_mol.name).lower().replace("_", "-"),
        ),
        fulleren_init,
        addon_mol,
        model_file="DPA2_medium_28_10M_rc0_MP_traj_v024_alldata_mixu.pth",
        addon_start=0,
        start_idx_list=[
            []
            # list(range(29))
        ],
        group_size=300,
        generate_addons_group_size=10,
        addon_max=max_addon_num,
        addon_step=1,
        pick_first_n=100,
        # addon_bond_length=1.4
        gpu_machine_template_config=gpu_machine_template_config,
        reuse_step=reuse_step_list,
    )
