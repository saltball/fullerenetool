from pathlib import Path
from typing import List

import fpop
from dflow.python import OP, Artifact, BigParameter, upload_packages
from dflow.utils import run_command as dflow_run_command
from dflow.utils import set_directory

from fullerenetool.logger import logger


class Cp2kInputs:
    def __init__(self, inp_file: str):
        """
        Initialize the Cp2kInputs class.

        Parameters
        ----------
        inp_file : str
            The path to the user-submitted CP2K input file.
        """
        self.inp_file_from_file(inp_file)

    @property
    def inp_template(self):
        """
        Return the template content of the input file.
        """
        return self._inp_template

    def inp_file_from_file(self, fname: str):
        """
        Read the content of the input file and store it.

        Parameters
        ----------
        fname : str
            The path to the input file.
        """
        self._inp_template = Path(fname).read_text()


upload_packages += fpop.__path__


def generate_cp2k_input(
    template_path=None,
):
    return Cp2kInputs(
        inp_file=template_path,
    )


@OP.function
def prepCp2kCalculation(
    calculation_system: BigParameter("dpdata.System"),
    input_obj: BigParameter(Cp2kInputs),
) -> {"input_dir": Artifact(List[Path]), "task_num": int, "task_name_list": List[str]}:
    result_dir = []
    task_num = len(calculation_system)
    task_name_list = []
    for work_idx, frame in enumerate(calculation_system):
        job_name = "cp2k_task_{:0{width}}".format(
            work_idx,
            width=len(str(task_num)) + 1,
        )
        work_dir = Path(job_name)
        # Generate coord.xyz file
        xyz_string = ""
        with set_directory(work_dir, mkdir=True):
            for atom_symb, atom_posi in zip(
                [frame.data["atom_names"][tt] for tt in frame.data["atom_types"]],
                frame.data["coords"][0],
            ):
                xyz_string += f"""{atom_symb} {float(atom_posi[0]):.6f} {
                    float(atom_posi[1]):.6f} {float(atom_posi[2]):.6f}\n"""
            Path("coord.xyz").write_text(xyz_string)
            # Generate CELL_PARAMETER file
            cell_params = frame["cells"][0]
            with open("CELL_PARAMETER", "w") as file:
                file.write(
                    f"""A {cell_params[0, 0]:14.8f} \
                        {cell_params[0, 1]:14.8f} {cell_params[0, 2]:14.8f}\n"""
                )
                file.write(
                    f"""B {cell_params[1, 0]:14.8f} \
                        {cell_params[1, 1]:14.8f} {cell_params[1, 2]:14.8f}\n"""
                )
                file.write(
                    f"""C {cell_params[2, 0]:14.8f} \
                        {cell_params[2, 1]:14.8f} {cell_params[2, 2]:14.8f}\n"""
                )
            # Write the CP2K input file content
            Path("input.inp").write_text(input_obj.inp_template)
        result_dir.append(work_dir)
        task_name_list.append(job_name)
    return {
        "input_dir": result_dir,
        "task_num": task_num,
        "task_name_list": task_name_list,
    }


@OP.function
def runCp2KCalculation(
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

    os.chdir(input_dir)
    time_log_path = Path(f"time_log.{job_name}")
    time_log_path.touch()
    command = " ".join([command, "-i", "input.inp", "-o", "log"])
    if time_benchmark:
        command = f"""echo START: $(date "+%Y-%m-%d %H:%M:%S")>> \
            {time_log_path.as_posix()} && {command} && echo END: \
            $(date "+%Y-%m-%d %H:%M:%S")>>{time_log_path.as_posix()}"""
    ret, out, err = dflow_run_command(
        command,
        try_bash=True,
        shell=True,
        interactive=False,
        print_oe=True,
        raise_error=True,
    )
    logger.error(err)
    return {
        "output_dir": Path(input_dir),
        "log": Path("log"),
        "time_log_path": Path(time_log_path),
    }
