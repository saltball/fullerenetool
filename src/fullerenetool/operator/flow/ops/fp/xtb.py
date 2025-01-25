import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Union

from dflow.python import OP, Artifact, BigParameter
from dflow.utils import run_command as dflow_run_command
from dflow.utils import set_directory

from fullerenetool.logger import logger

class XtbInputs:

    flag_runtypes = [
        "scc",
        "grad",
        "vip",
        "vea",
        "vipea",
        "vomega",
        "vfukui",
        "dipro",
        "esp",
        "stm",
        "opt",
        "metaopt",
        "hess",
        "md",
    ]
    def __init__(
        self,
        **kwargs,
    ):
        self._input_dict = kwargs
        
    
    def inp_template(self):
        inp_template_string = ""
        for key, value in self._input_dict.items():
            if value is not None:
                if isinstance(value, bool) or key in self.flag_runtypes:
                    inp_template_string += f"-{key}" if len(key) == 1 else f"--{key}"
                else:
                    inp_template_string += f"--{key} {value}" if len(key) > 1 else f"-{key} {value}"
        return inp_template_string



class prepXtbCalculation(OP):
    @classmethod
    def get_input_sign(cls):
        return {
            "calculation_system": BigParameter("dpdata.System"),
            "input_obj": BigParameter(XtbInputs, default=None),
        }

    @classmethod
    def get_output_sign(cls):
        return {
            "input_dir": Artifact(List[Path]),
            "task_num": int,
            "task_name_list": List[str]
        }

    @OP.exec_sign_check
    def execute(self, op_in):
        calculation_system = op_in["calculation_system"]
        input_obj = op_in["input_obj"]
        import shutil

        result_dir = []
        task_num = len(calculation_system)
        task_name_list = []
        for work_idx, frame in enumerate(calculation_system):
            job_name = "xtb_task_{:0{width}}".format(
                work_idx,
                width=len(str(task_num)) + 1,
            )
            work_dir = Path(job_name)
            with set_directory(work_dir, mkdir=True):
                xyz_string = ""
                xyz_string += f"{len(frame.data['atom_types'])}\n\n"
                for atom_symb, atom_posi in zip(
                    [frame.data["atom_names"][tt] for tt in frame.data["atom_types"]],
                    frame.data["coords"][0],
                ):
                    xyz_string += f"""{atom_symb} {float(atom_posi[0]):.6f} {
                        float(atom_posi[1]):.6f} {float(atom_posi[2]):.6f}\n"""
                Path("{}.xyz".format(job_name)).write_text(xyz_string)
                Path("input_args").write_text(input_obj.inp_template())
            result_dir.append(work_dir)
            task_name_list.append(job_name)
        return {
            "input_dir": result_dir,
            "task_num": task_num,
            "task_name_list": task_name_list,
        }


@OP.function
def runXtbCalculation(
    command: str,
    job_name: str,
    input_dir: Artifact(Path),
    time_benchmark: bool,
) -> {
    "output_dir": Artifact(Path),
    "log": Artifact(Path),
    "time_log_path": Artifact(Path),
}:
    import os
    from pathlib import Path

    with set_directory(input_dir, mkdir=False):
        time_log_path = Path(f"time_log.{job_name}")
        time_log_path.touch()
        command = f"{command} {job_name}.xyz {Path('input_args').read_text()}"
        if time_benchmark:
            command = f"""echo START: $(date "+%Y-%m-%d %H:%M:%S")>>\
                {time_log_path.as_posix()} && {command} && echo END: \
                $(date "+%Y-%m-%d %H:%M:%S")>>{time_log_path.as_posix()}"""
        ret, out, err = dflow_run_command(
            command,
            try_bash=True,
            shell=True,
            interactive=False,
            raise_error=True,
        )
        Path("log").write_text(out)
    return {
        "output_dir": input_dir,
        "log": input_dir / "log",
        "time_log_path": time_log_path,
    }