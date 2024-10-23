from pathlib import Path
from typing import List, Optional

from dflow.python import OP, Artifact, BigParameter
from dflow.utils import run_command as dflow_run_command
from dflow.utils import set_directory


class OrcaInputGenerator:
    def __init__(
        self, calculate_system, *, keywords: str, charge: int, multiplicity: int
    ):
        self.keywords = keywords
        self.charge = charge
        self.multiplicity = multiplicity
        self.moinp = None
        self.maxcore = 4096
        self.basename = None
        self.calculate_system = calculate_system

    def set_moinp(self, moinp: str):
        self.moinp = moinp

    def set_maxcore(self, maxcore: int):
        self.maxcore = maxcore

    def set_basename(self, basename: str):
        self.basename = basename

    def build_input(self):
        input_content = f"! {self.keywords}\n"

        if self.moinp:
            input_content += "! moread\n"
            input_content += f'%moinp "{self.moinp}"\n'

        input_content += f"%maxcore {self.maxcore}\n"

        if self.basename:
            input_content += f'%base "{self.basename}"\n'

        input_content += f"* xyz {self.charge} {self.multiplicity}\n"

        # Write coordinates
        atom_names = self.calculate_system["atom_names"]
        atom_types = self.calculate_system["atom_types"]

        for atom_type, positions in zip(
            [atom_names[atom_type] for atom_type in atom_types],
            self.calculate_system["coords"][0],
        ):
            input_content += (
                f"{atom_type} "
                + f"{positions[0]:f} {positions[1]:f} {positions[2]:f}\n"
            )

        input_content += "*\n"

        return input_content

    def write_input(self, file_name):
        input_content = self.build_input()
        with open(file_name, "w") as f:
            f.write(input_content)


def generate_orca_input(
    calculate_system,
    *,
    keywords: str,
    moinp: Optional[str] = None,
    maxcore: int = 4096,
    basename: Optional[str] = None,
    charge: int = 0,
    multiplicity: int = 1,
):
    """
    Creates and returns a configured ORCA input file generator.

    Parameters:
    - keywords (str): Keywords for the ORCA calculation.
    - moinp (Optional[str]): If provided, used as the MO initialization file.
    - maxcore (int): Maximum core memory allocated to the ORCA calculation in MB.
    - basename (Optional[str]): Base name for the ORCA output files.
    - charge (int): Charge of the computational system.
    - multiplicity (int): Multiplicity of the computational system.

    Returns:
    - An instance of OrcaInputGenerator configured with the given parameters.
    """
    # Create an ORCA input file generator instance
    generator = OrcaInputGenerator(
        calculate_system,
        keywords=keywords,
        charge=charge,
        multiplicity=multiplicity,
    )

    # Set the MO initialization file if provided
    if moinp:
        generator.set_moinp(moinp)

    # Set the maximum core memory
    generator.set_maxcore(maxcore)

    # Set the base name for output files if provided
    if basename:
        generator.set_basename(basename)

    # Return the configured generator instance
    return generator


@OP.function
def prepOrcaCalculation(
    calculation_system: BigParameter("dpdata.System"),
    input_obj: BigParameter(OrcaInputGenerator),
) -> {"input_dir": Artifact(List[Path]), "task_num": int, "task_name_list": List[str]}:
    result_dir = []
    task_num = len(calculation_system)
    task_name_list = []
    for work_idx, calculation_sys in enumerate(calculation_system):
        job_name = "orca_task_{:0{width}}".format(
            work_idx,
            width=len(str(task_num)) + 1,
        )
        work_dir = Path(job_name)
        with set_directory(work_dir, mkdir=True):
            input_obj.write_input(
                file_name=f"{job_name}.inp",
            )
        result_dir.append(work_dir)
        task_name_list.append(job_name)
    return {
        "input_dir": result_dir,
        "task_num": task_num,
        "task_name_list": task_name_list,
    }


@OP.function
def runOrcaCalculation(
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

    from fullerenetool.logger import logger

    os.chdir(input_dir)
    time_log_path = Path(f"time_log.{job_name}")
    time_log_path.touch()
    command = " ".join(['bash -lic "', command, f"{job_name}.inp", '"', "| tee log"])
    if time_benchmark:
        command = f"""echo START: $(date "+%Y-%m-%d %H:%M:%S")>>\
            {time_log_path.as_posix()} && {command} && echo END: \
            $(date "+%Y-%m-%d %H:%M:%S")>>{time_log_path.as_posix()}"""
    ret, out, err = dflow_run_command(
        command,
        try_bash=True,
        shell=True,
        interactive=True,
        raise_error=True,
    )
    logger.info("out:{}".format(out))
    logger.warning("err:{}".format(err))
    return {
        "output_dir": Path(input_dir),
        "log": Path("log"),
        "time_log_path": Path(time_log_path),
    }
