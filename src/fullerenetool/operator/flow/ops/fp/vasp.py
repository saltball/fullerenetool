from pathlib import Path
from typing import List

import fpop
from dflow.python import OP, Artifact, BigParameter, upload_packages
from dflow.utils import run_command as dflow_run_command
from dflow.utils import set_directory

from fullerenetool.logger import logger

upload_packages += fpop.__path__


def generate_vasp_input(
    input_path,
    *,
    ncore,
    kspacing,
    kpar=None,
    npar=None,
    kgamma=False,
    ispin=1,
    ismear=0,
    sigma=0.05,
    lorbit=11,
    lmaxmix=2,
    encut=500,
    ediff=1.000000e-06,
    pp_files=None,
    ion_relax=None,
    magmom=None,
    nelm=100,
    **kwargs,
):
    from fpop.vasp import VaspInputs

    vasp_input_str = f"""#Initialize
ISTART = 1
ICHARG = 2
ISYM = 0
PREC = Accurate
ALGO = Normal # or fast

#Electronic degrees
ENCUT = {encut}
NELM    =  {nelm}
NELMIN  =  6
EDIFF = {ediff}

#iron:1 0.2 non-iron:0 0.05###################
ISMEAR = {ismear}
SIGMA = {sigma}

# use for magmom or +U
ISPIN = {ispin}
LMAXMIX = {lmaxmix}

LASPH = .TRUE.
LORBIT = {lorbit}

#File to write
LCHARG=F
LWAVE=F

#Make it faster
NCORE = {ncore}
NPAR = {npar}
LREAL=A
"""
    if kpar is not None:
        if kgamma and kpar != 1:
            raise ValueError("kgamma and kpar cannot be set at the same time")
        vasp_input_str += f"KPAR = {kpar}\nKGAMMA = False\n"
    else:
        if kgamma:
            vasp_input_str += "KGAMMA = True\n"
    if ion_relax is not None:
        vasp_input_str += f"""#Ionic parameters
IBRION = 2
NSW = {ion_relax}
"""
    else:
        vasp_input_str += """#Ionic parameters
IBRION = -1  # 2
NSW = 0
"""
    if magmom is not None:
        vasp_input_str += "MAGMOM = " + " ".join([str(m) for m in magmom]) + "\n"
    for key, value in kwargs.items():
        vasp_input_str += "{}={}\n".format(key, value)
    Path(input_path).write_text(vasp_input_str)
    logger.info("vasp input file is written to {}".format(input_path))
    vasp_inputs = VaspInputs(
        kspacing,
        Path(input_path),
        pp_files,
        kgamma=kgamma,
    )
    return vasp_inputs


@OP.function
def prepVaspCalculation(
    calculation_system: BigParameter("dpdata.System"),
    input_obj: BigParameter("fpop.vasp.VaspInputs"),
    common_optional_files: Artifact(List[Path], optional=True),
) -> {"input_dir": Artifact(List[Path]), "task_num": int, "task_name_list": List[str]}:
    import random
    import string

    import dpdata
    from pymatgen.core.structure import Structure
    from pymatgen.io.vasp.sets import MPStaticSet

    input_obj: fpop.vasp.VaspInputs = input_obj
    result_dir = []
    random_string = "".join(
        random.choice(string.ascii_lowercase + string.digits) for _ in range(8)
    )
    if isinstance(calculation_system, dpdata.System):
        formula = calculation_system.short_formula
        calculation_system = [frame for frame in calculation_system]  # convert to list
    else:
        formula = "multi"
    task_name_list = []
    task_num = len(calculation_system)
    for work_idx, calculation_sys in enumerate(calculation_system):
        job_name = "vasp_task_{formula}_{random_string}_{work_idx:0{width}}".format(
            formula=formula,
            random_string=random_string,
            work_idx=work_idx,
            width=len(str(task_num)) + 1,
        )
        work_dir = Path(job_name)
        with set_directory(work_dir, mkdir=True):
            calculation_sys.to("vasp/poscar", "POSCAR")
            # write with pymatgen
            a = Structure.from_file("POSCAR")
            b = MPStaticSet(a)
            b.write_input(".", potcar_spec=True)
            # overwrite incar
            Path("INCAR").write_text(input_obj.incar_template)
            # avoid some element have 0 atom by multisystem
            tmp_frame = dpdata.System("POSCAR", fmt="vasp/poscar")
            Path("POTCAR").write_text(input_obj.make_potcar(tmp_frame["atom_names"]))
            Path("KPOINTS").write_text(
                input_obj.make_kpoints(calculation_sys["cells"][0])
            )
            if common_optional_files:
                import shutil

                for file in common_optional_files:
                    shutil.copy(file, ".")
        result_dir.append(work_dir)
        task_name_list.append(job_name)
    return {
        "input_dir": result_dir,
        "task_num": task_num,
        "task_name_list": task_name_list,
    }


@OP.function
def runVaspCalculation(
    command: str,
    job_name: str,
    input_dir: Artifact(Path),
    time_benchmark: bool,
) -> {
    "output_dir": Artifact(Path),
    "log": Artifact(Path),
    "time_log_path": Artifact(Path),
}:
    result_files = [
        "OUTCAR",
        "vasprun.xml",
        "KPOINTS",
        "CONTCAR",
        "OSZICAR",
        "CHGCAR",
        "WAVECAR",
        "stdlog",
        "POSCAR",
        f"{job_name}.log",
        f"time_log.{job_name}",
    ]
    import os
    import shutil
    from pathlib import Path

    os.chdir(input_dir)
    time_log_path = Path(f"time_log.{job_name}")
    time_log_path.touch()
    command = f"ulimit -s unlimited >> {job_name}.log && {command} >> {job_name}.log"
    if time_benchmark:
        command = f"""echo START: $(date "+%Y-%m-%d %H:%M:%S")>>\
            {time_log_path.as_posix()} && {command}\
            && echo END: $(date "+%Y-%m-%d %H:%M:%S")>>{time_log_path.as_posix()}"""
    ret, out, err = dflow_run_command(
        command,
        try_bash=True,
        shell=True,
        interactive=False,
        raise_error=False,
    )
    Path(job_name).mkdir(parents=True, exist_ok=True)
    for file in result_files:
        if Path(file).exists() and Path(file).is_file():
            shutil.copy(file, Path(job_name, file))
    return {
        "output_dir": Path(job_name),
        "log": Path(f"{job_name}.log"),
        "time_log_path": Path(time_log_path),
    }
