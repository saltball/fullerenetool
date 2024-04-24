from pathlib import Path
from typing import List

from dflow import (
    InputParameter,
    Inputs,
    OutputArtifact,
    OutputParameter,
    Outputs,
    Step,
    Steps,
    argo_range,
)
from dflow.python import OP, Artifact, PythonOPTemplate, Slices


@OP.function
def time2second(
    time_log_path: Artifact(List[Path]), price_data: float
) -> {"time_second": float, "price_cost": float}:
    import re
    from datetime import datetime

    import numpy as np

    price_cost_list = []
    time_second_list = []
    for time_log_path_item in time_log_path:
        with open(time_log_path_item, "r") as f:
            lines = f.readlines()
        start_time = re.findall(r"START: (.*)", lines[0])[0]
        end_time = re.findall(r"END: (.*)", lines[1])[0]
        TIME_FORMAT = r"%Y-%m-%d %H:%M:%S"
        time_difference = datetime.strptime(end_time, TIME_FORMAT) - datetime.strptime(
            start_time, TIME_FORMAT
        )
        price_cost = float(time_difference.total_seconds()) / 3600 * price_data
        price_cost_list.append(price_cost)
        time_second_list.append(time_difference.total_seconds())
    return {
        "time_second": float(np.mean(time_second_list)),
        "price_cost": float(np.mean(price_cost_list)),
    }


def CalculationSteps(
    *,
    name,
    prepStep,
    runStep,
    run_command,
    result_convert=None,
    group_size=5,
    time_benchmark=True,
    executor=None,
    prep_image=None,
    run_image=None,
    time_benchmark_image=None,
    run_template_config=None,
    other_template_config=None,
):
    name = name.lower()
    calculation_steps = Steps(
        name=f"calculation-steps-{name}",
        inputs=Inputs(
            parameters=(
                {
                    "calculation_systems": InputParameter(),
                    "input_obj": InputParameter(),
                    "price_data": InputParameter(),
                }
                if time_benchmark
                else {
                    "calculation_systems": InputParameter(),
                    "input_obj": InputParameter(),
                }
            )
        ),
        outputs=Outputs(
            parameters=(
                {
                    "time_second": OutputParameter(),
                }
                if time_benchmark
                else {}
            ),
        ),
    )
    pre_step = Step(
        name=f"prep-run-{name}",
        template=PythonOPTemplate(prepStep, image=prep_image, **other_template_config),
        parameters={
            "calculation_system": calculation_steps.inputs.parameters[
                "calculation_systems"
            ],
            "input_obj": calculation_steps.inputs.parameters["input_obj"],
        },
        key=f"prep-run-{name}",
    )
    calculation_steps.add(pre_step)
    run_step = Step(
        name=f"run-{name}",
        template=PythonOPTemplate(
            runStep,
            image=run_image,
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
            ),
            **run_template_config,
        ),
        parameters={
            "command": run_command,
            "job_name": pre_step.outputs.parameters["task_name_list"],
            "time_benchmark": time_benchmark,
        },
        artifacts={
            "input_dir": pre_step.outputs.artifacts["input_dir"],
        },
        key=f"run-{name}",
        executor=executor,
        with_param=argo_range(pre_step.outputs.parameters["task_num"]),
    )
    calculation_steps.add(run_step)
    additional_steps = []
    if time_benchmark:
        time2second_step = Step(
            name=f"time2second-{name}",
            template=PythonOPTemplate(
                time2second, image=time_benchmark_image, **other_template_config
            ),
            parameters=(
                {
                    "price_data": calculation_steps.inputs.parameters["price_data"],
                }
                if time_benchmark
                else {}
            ),
            artifacts={
                "time_log_path": run_step.outputs.artifacts["time_log_path"],
            },
            key=f"time2second-{name}",
        )
        additional_steps.append(time2second_step)

    if time_benchmark:
        calculation_steps.outputs.parameters["time_second"].value_from_parameter = (
            time2second_step.outputs.parameters["time_second"]
        )
    if result_convert is not None:
        result_convert_step = Step(
            name=f"result-convert-{name}",
            template=PythonOPTemplate(
                result_convert, image=time_benchmark_image, **other_template_config
            ),
            artifacts={
                "output_dir": run_step.outputs.artifacts["output_dir"],
            },
            key=f"result-convert-{name}",
        )
        additional_steps.append(result_convert_step)
    calculation_steps.add(additional_steps)
    calculation_steps.outputs.artifacts["output_dir"] = OutputArtifact(
        _from=run_step.outputs.artifacts["output_dir"]
    )
    calculation_steps.outputs.artifacts["log"] = OutputArtifact(
        _from=run_step.outputs.artifacts["log"]
    )
    calculation_steps.outputs.artifacts["time_log_path"] = OutputArtifact(
        _from=run_step.outputs.artifacts["time_log_path"]
    )

    return calculation_steps
