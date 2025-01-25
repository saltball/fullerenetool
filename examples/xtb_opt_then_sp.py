from pathlib import Path
import tempfile
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
from dflow.python import OP, PythonOPTemplate, Slices, upload_packages, BigParameter, Artifact
import fullerenetool
from fullerenetool.operator.flow.ops.fp.xtb import runXtbCalculation, prepXtbCalculation, XtbInputs

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

UTILIMAGE = "registry-vpc.cn-heyuan.aliyuncs.com/xjtu-icp/fullerenetool:2024-10-24-00-38-18"
XTBIMAGE = "registry-vpc.cn-heyuan.aliyuncs.com/xjtu-icp/xtb:latest"


def generate_scan_system_list(
    symbol1: str,
    symbol2: str,
    distance_list: List[float],
):
    from dpdata.plugins.ase import ASEStructureFormat
    from dpdata import System
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
)->{
    "result": BigParameter(List[dpdata.System])
}:
    from ase.io.extxyz import read_xyz
    from dpdata.plugins.ase import ASEStructureFormat
    system_list = []
    for xtb_result in xtb_results:
        atoms = list(read_xyz((Path(xtb_result)/"xtbopt.xyz").open('r'), index=0))[-1]
        system = dpdata.System(data=ASEStructureFormat().from_system(atoms))
        system_list.append(system)
        perturb_sys = system.perturb(
            pert_num=pert_num,
            atom_pert_distance=atom_pert_distance,
            cell_pert_fraction=0,
        )
        for frame in perturb_sys:
            system_list.append(frame)
    return {
        "result": system_list
    }


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
                group_size=1,
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
                group_size=20,
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
    scan_system = generate_scan_system_list(
        "C",
        "C",
        list(np.linspace(1.0, 1.5, 11))
    )
    xtb_opt_input = XtbInputs(opt=True)
    xtb_sp_input = XtbInputs(grad=True)
    with tempfile.TemporaryDirectory() as tmp_dir:
        run_main(
            flow_name="xtb-scan",
            calculation_system=scan_system,
            opt_input=xtb_opt_input,
            sp_input=xtb_sp_input,
            run_command="xtb",
            pert_num=5,
            atom_pert_distance=0.1,
            run_machine_template_config=run_machine_template_config_c8,
        )
