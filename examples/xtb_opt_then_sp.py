import tempfile
from pathlib import Path
from typing import List

import ase
import dpdata
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
from dflow.python import (
    OP,
    Artifact,
    BigParameter,
    PythonOPTemplate,
    Slices,
    upload_packages,
)

import fullerenetool
from fullerenetool.operator.flow.ops.fp.xtb import (
    XtbInputs,
    prepXtbCalculation,
    runXtbCalculation,
)

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

run_machine_template_config_c8 = {
    "annotations": {
        "k8s.aliyun.com/eci-use-specs": "ecs.e-c1m2.2xlarge",  # 8C16G
        "k8s.aliyun.com/eci-spot-strategy": "SpotWithPriceLimit",  # 采用系统自动出价，跟随当前市场实际价格
        "k8s.aliyun.com/eci-spot-price-limit": "0.1",  # 设置最高出价
        "k8s.aliyun.com/eci-spot-duration": "0",  # 设置无保护期
        "k8s.aliyun.com/eci-spot-release-strategy": "api-evict",  # 释放策略
        "k8s.aliyun.com/eci-fail-strategy": "fail-back",
    }
}

run_machine_template_config_c16 = {
    "annotations": {
        "k8s.aliyun.com/eci-use-specs": "ecs.u1-c1m2.4xlarge",  # 16C32G
        "k8s.aliyun.com/eci-spot-strategy": "SpotWithPriceLimit",  # 采用系统自动出价，跟随当前市场实际价格
        "k8s.aliyun.com/eci-spot-price-limit": "0.3",  # 设置最高出价
        "k8s.aliyun.com/eci-spot-duration": "0",  # 设置无保护期
        "k8s.aliyun.com/eci-spot-release-strategy": "api-evict",  # 释放策略
        "k8s.aliyun.com/eci-fail-strategy": "fail-back",
    }
}

run_machine_template_config = run_machine_template_config_c8

UTILIMAGE = (
    "registry-vpc.cn-heyuan.aliyuncs.com/xjtu-icp/fullerenetool:2024-10-24-00-38-18"
)
XTBIMAGE = "registry-vpc.cn-heyuan.aliyuncs.com/xjtu-icp/xtb:latest"


def generate_scan_system_list(
    symbol1: str,
    symbol2: str,
    distance_list: List[float],
):
    from dpdata import System
    from dpdata.plugins.ase import ASEStructureFormat

    system_list = []
    for distance in distance_list:
        calculate_system = System(
            data=ASEStructureFormat().from_system(
                ase.Atoms(
                    symbols=[symbol1, symbol2],
                    positions=[[0, 0, 0], [0, 0, distance]],
                    cell=[[15, 0, 0], [0, 15, 0], [0, 0, 20]],
                )
            )
        )
        system_list.append(calculate_system)
    return system_list


@OP.function
def xtb_opt_results_to_system(
    xtb_results: Artifact(List[Path]),
    pert_num: float,
    atom_pert_distance: float,
) -> {"result": BigParameter(List[dpdata.System])}:
    from ase.io.extxyz import read_xyz
    from dpdata.plugins.ase import ASEStructureFormat

    system_list = []
    for xtb_result in xtb_results:
        atoms = list(read_xyz((Path(xtb_result) / "xtbopt.xyz").open("r"), index=0))[-1]
        system = dpdata.System(data=ASEStructureFormat().from_system(atoms))
        system_list.append(system)
        perturb_sys = system.perturb(
            pert_num=pert_num,
            atom_pert_distance=atom_pert_distance,
            cell_pert_fraction=0,
        )
        for frame in perturb_sys:
            system_list.append(frame)
    return {"result": system_list}


def run_main(
    flow_name: str,
    calculation_system: List[dpdata.System],
    opt_input: XtbInputs,
    sp_input: XtbInputs,
    run_command: str,
    pert_num=10,
    atom_pert_distance=0.2,
    simple_machine_template_config=simple_machine_template_config,
    run_machine_template_config=run_machine_template_config,
):
    wf = Workflow(name=flow_name)
    # opt
    prep_xtb_opt_calculation = Step(
        name="prep-xtb-opt-calculation",
        template=PythonOPTemplate(
            prepXtbCalculation,
            image=UTILIMAGE,
            **simple_machine_template_config,
        ),
        parameters={
            "calculation_system": calculation_system,
            "input_obj": opt_input,
        },
        key=f"prep-opt-{flow_name}",
    )
    wf.add(prep_xtb_opt_calculation)
    run_xtb_opt_calculation = Step(
        name="run-xtb-opt-calculation",
        template=PythonOPTemplate(
            runXtbCalculation,
            image=XTBIMAGE,
            slices=Slices(
                input_parameter=[
                    "job_name",
                ],
                input_artifact=[
                    "input_dir",
                ],
                output_artifact=[
                    "output_dir",
                    "log",
                    "time_log_path",
                ],
                group_size=5,
                pool_size=1,
            ),
            **run_machine_template_config,
        ),
        parameters={
            "command": run_command,
            "job_name": prep_xtb_opt_calculation.outputs.parameters["task_name_list"],
            "time_benchmark": True,
        },
        artifacts={
            "input_dir": prep_xtb_opt_calculation.outputs.artifacts["input_dir"],
        },
        key=f"run-xtb-opt-calculation-{flow_name}",
        with_sequence=argo_sequence(
            argo_len(prep_xtb_opt_calculation.outputs.parameters["task_name_list"])
        ),
    )
    wf.add(run_xtb_opt_calculation)

    # gather opt results
    gather_xtb_opt_results = Step(
        name="gather-xtb-opt-results",
        template=PythonOPTemplate(
            xtb_opt_results_to_system,
            image=UTILIMAGE,
            **simple_machine_template_config,
        ),
        parameters={
            "pert_num": pert_num,
            "atom_pert_distance": atom_pert_distance,
        },
        artifacts={
            "xtb_results": run_xtb_opt_calculation.outputs.artifacts["output_dir"],
        },
        key=f"gather-opt-results-{flow_name}",
    )
    wf.add(gather_xtb_opt_results)

    # sp
    prep_xtb_calculation = Step(
        name="prep-xtb-calculation",
        template=PythonOPTemplate(
            prepXtbCalculation,
            image=UTILIMAGE,
            **simple_machine_template_config,
        ),
        parameters={
            "calculation_system": gather_xtb_opt_results.outputs.parameters["result"],
            "input_obj": sp_input,
        },
        key=f"prep-run-{flow_name}",
    )
    wf.add(prep_xtb_calculation)
    run_xtb_calculation = Step(
        name="run-xtb-calculation",
        template=PythonOPTemplate(
            runXtbCalculation,
            image=XTBIMAGE,
            slices=Slices(
                input_parameter=[
                    "job_name",
                ],
                input_artifact=[
                    "input_dir",
                ],
                output_artifact=[
                    "output_dir",
                    "log",
                    "time_log_path",
                ],
                group_size=100,
                pool_size=1,
            ),
            **run_machine_template_config,
        ),
        parameters={
            "command": run_command,
            "job_name": prep_xtb_calculation.outputs.parameters["task_name_list"],
            "time_benchmark": True,
        },
        artifacts={
            "input_dir": prep_xtb_calculation.outputs.artifacts["input_dir"],
        },
        key=f"run-xtb-calculation-{flow_name}",
        with_sequence=argo_sequence(
            argo_len(prep_xtb_calculation.outputs.parameters["task_name_list"])
        ),
    )
    wf.add(run_xtb_calculation)
    wf.submit()


if __name__ == "__main__":
    import itertools
    import os
    import random
    from pathlib import Path

    import pynauty as pn
    from ase.visualize import view
    from dpdata import System
    from dpdata.plugins.ase import ASEStructureFormat

    from fullerenetool.fullerene.cage import FullereneCage
    from fullerenetool.fullerene.derivatives import (
        DerivativeFullereneGraph,
        DerivativeGroup,
        addons_to_fullerene,
    )

    def random_refuse_combinations(
        cage_num: int,
        addon_num: int,
        pick_num: int,
    ):
        import math

        total_comb = math.comb(cage_num, addon_num)
        accept_prob = 0.01 if pick_num / total_comb < 0.01 else pick_num / total_comb
        print(f"accept_prob:{accept_prob}")
        selected_comb = set()

        for comb in itertools.combinations(range(cage_num), addon_num):
            if np.random.random() < accept_prob:
                selected_comb.add(tuple(sorted(comb)))
                print(comb)

            if len(selected_comb) >= pick_num:
                break
        return list(selected_comb)

    from fullerenetool.operator.graph import nx_to_nauty

    os.chdir(Path(__file__).parent)
    from ase.io.extxyz import read_extxyz, write_extxyz

    c70 = list(read_extxyz(Path("fullerene_xyz/C70_000008149opt.xyz").open("r")))[-1]
    c20 = list(read_extxyz(Path("fullerene_xyz/C20_000000001opt.xyz").open("r")))[-1]
    c80 = list(read_extxyz(Path("fullerene_xyz/C80_000031924opt.xyz").open("r")))[-1]
    for fullerene_atoms, add_num, pick_num, dev_atom in zip(
        [
            # c70,
            c70,
            c70,
            c70,
            c70,
            c70,
            c70,
            c70,
            c70,
        ],
        [
            # 2,
            10,
            20,
            2,
            10,
            20,
            2,
            10,
            20,
        ],
        [
            # 200,
            200,
            200,
            200,
            200,
            200,
            200,
            200,
            200,
        ],
        [
            # 'H',
            "H",
            "H",
            "F",
            "F",
            "F",
            "Cl",
            "Cl",
            "Cl",
        ],
    ):

        # fullerene_atoms = list(read_extxyz(
        #     Path("fullerene_xyz/C70_000008149opt.xyz").open('r')))[-1]
        # add_num = 2
        # pick_num = 200
        # dev_atom= 'H'

        addon_mol = DerivativeGroup(
            atoms=ase.Atoms(
                "X{}".format(dev_atom),
                [
                    [0.5, 0.5, 0.0],
                    [0, 0, 0],
                    #  [0, 0, 1.4]
                ],
            ),
            graph=nx.from_numpy_array(
                np.array(
                    [
                        [
                            0,
                            1,
                            #  0
                        ],
                        [
                            1,
                            0,
                            #  1
                        ],
                        # [0, 1, 0]
                    ]
                )
            ),
            addon_atom_idx=0,
        )

        addons_list = [addon_mol] * add_num

        fulleren_init = FullereneCage(fullerene_atoms)

        canon_index = pn.canon_label(
            nx_to_nauty(fulleren_init.graph.graph, include_z_labels=False)
        )

        fulleren_init = FullereneCage(fullerene_atoms[np.array(canon_index)])

        calculation_system_list = []
        add_index_list_list = random_refuse_combinations(
            len(fullerene_atoms),
            addon_num=add_num,
            pick_num=pick_num,
        )
        print(add_index_list_list)
        for add_index_list in add_index_list_list:
            dev_graph, dev_fullerenes = addons_to_fullerene(
                addons_list,
                add_index_list,
                fulleren_init,
                nx.adjacency_matrix(fulleren_init.graph.graph).todense(),
            )
            calculation_system_list.append(
                System(data=ASEStructureFormat().from_system(dev_fullerenes))
            )

        xtb_opt_input = XtbInputs(opt=True)
        xtb_sp_input = XtbInputs(grad=True)
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_main(
                flow_name="xtb-fullerene-dev-c{}-{}{}".format(
                    len(fullerene_atoms), dev_atom.lower(), add_num
                ),
                calculation_system=calculation_system_list,
                opt_input=xtb_opt_input,
                sp_input=xtb_sp_input,
                run_command="xtb",
                pert_num=10,
                atom_pert_distance=0.2,
                run_machine_template_config=run_machine_template_config_c8,
            )
