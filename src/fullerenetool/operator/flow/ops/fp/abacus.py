from pathlib import Path
from typing import List

import fpop
from dflow.python import OP, Artifact, BigParameter, upload_packages
from dflow.utils import run_command as dflow_run_command
from dflow.utils import set_directory
from fpop.abacus import AbacusInputs

upload_packages += fpop.__path__


def generate_abacus_input(
    input_path,
    *,
    ntype,
    job_name,
    kspacing,  # in vasp unit, abacus use Bohr but vasp use Angstrom
    ecutwfc=50.0,
    scf_thr=1.000000e-08,
    basis_type="pw",
    upf_map: dict = {"C": "C_ONCV_PBE-1.0.upf"},
):
    Path(input_path).write_text(
        f"""INPUT_PARAMETERS
calculation scf
suffix {job_name}
ntype {ntype}
symmetry 0
ecutwfc {ecutwfc}
scf_thr {scf_thr}
scf_nmax 100
cal_force 1
cal_stress 1
basis_type {basis_type}
smearing_method mp
smearing_sigma 0.014600
mixing_type pulay
mixing_beta 0.700000
ks_solver dav
kspacing {kspacing/1.8897261246257702}"""
    )
    abacus_inputs = AbacusInputs(Path(input_path), upf_map)
    return abacus_inputs


@OP.function
def prepAbacusCalculation(
    calculation_system: BigParameter("dpdata.System"),
    input_obj: BigParameter(AbacusInputs),
) -> {"input_dir": Artifact(List[Path]), "task_num": int, "task_name_list": List[str]}:
    result_dir = []
    task_num = len(calculation_system)
    task_name_list = []
    for work_idx, calculation_sys in enumerate(calculation_system):
        job_name = "abacus_task_{:0{width}}".format(
            work_idx,
            width=len(str(task_num)) + 1,
        )
        work_dir = Path(job_name)
        with set_directory(work_dir, mkdir=True):
            element_list = calculation_sys["atom_names"]
            pp, orb = input_obj.write_pporb(element_list)
            dpks = input_obj.write_deepks()
            mass = input_obj.get_mass(element_list)
            calculation_sys.to(
                "abacus/stru",
                "STRU",
                pp_file=pp,
                numerical_orbital=orb,
                numerical_descriptor=dpks,
                mass=mass,
            )

            input_obj.write_input("INPUT")
            input_obj.write_kpt("KPT")
        result_dir.append(work_dir)
        task_name_list.append(job_name)
    return {
        "input_dir": result_dir,
        "task_num": task_num,
        "task_name_list": task_name_list,
    }


@OP.function
def runAbacusCalculation(
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
    if time_benchmark:
        command = f"""echo START: $(date "+%Y-%m-%d %H:%M:%S")>>\
            {time_log_path.as_posix()} && {command} | tee log && \
            echo END: $(date "+%Y-%m-%d %H:%M:%S")>>{time_log_path.as_posix()}"""
    ret, out, err = dflow_run_command(
        command, try_bash=True, shell=True, interactive=False
    )
    return {
        "output_dir": Path(f"OUT.{job_name}"),
        "log": Path("log"),
        "time_log_path": Path(time_log_path),
    }
