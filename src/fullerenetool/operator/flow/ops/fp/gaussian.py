import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

from dflow.python import OP, Artifact, BigParameter
from dflow.utils import run_command as dflow_run_command
from dflow.utils import set_directory


@dataclass
class GaussianLink0:
    r"""GaussianLink0 class, used for % command in the input file

    refer to https://gaussian.com/link0/
    """

    mem: Optional[str] = field(default=None)
    chk: Optional[str] = field(default=None)
    old_chk: Optional[str] = field(default=None)
    s_chk: Optional[str] = field(default=None)
    rwf: Optional[list] = field(default_factory=list)
    old_matrix: Optional[str] = field(default=None)
    old_raw_matrix: Optional[str] = field(default=None)
    int_file: Optional[str] = field(default=None)
    d2e_file: Optional[str] = field(default=None)
    kjob: Optional[str] = field(default=None)
    save: Optional[bool] = field(default=None)
    error_save: Optional[bool] = field(default=None)
    subst: Optional[str] = field(default=None)

    cpu: Optional[str] = field(default=None)
    nprocShared: Optional[int] = field(default=None)
    nproc: Optional[int] = field(default=None)

    def validate_mem(self):
        pattern = r"(\d+[KMGT]?[BW]?)"
        if not re.match(pattern, self.mem):
            raise ValueError("Invalid memory specification: {self.mem}")

    def validate_chk(self):
        pattern = r"(\S+).chk"
        if not re.match(pattern, self.chk):
            raise ValueError(f"Invalid specification for checkpoint: {self.chk}")

    def read_link0_commands(self, input_text):
        lines = input_text.strip().split("\n")
        for line in lines:
            if line.lower().startswith(r"%mem="):
                self.mem = line[5:].strip()
                self.validate_mem()
            elif line.lower().startswith(r"%chk="):
                self.chk = line[5:].strip()
                self.validate_chk()
            elif line.lower().startswith(r"%nproc="):
                self.nproc = int(line[7:].strip())
            elif line.lower().startswith(r"%cpu="):
                self.cpu = line[6:].strip().split()
            elif line.lower().startswith(r"%oldchk="):
                self.old_chk = line[8:].strip()
            elif line.lower().startswith(r"%savechk="):
                self.s_chk = line[9:].strip()

    def generate_link0_commands(self):
        commands = []
        for key, value in asdict(self).items():
            if isinstance(value, list):
                if len(value) > 0:
                    value = ",".join(map(str, value))
                    commands.append(f"%{key}={value}")
            elif value is not None:
                commands.append(f"%{key}={value}")
        return "\n".join(commands)


class GaussianInputBlock:
    """GaussianInputBlock class

    Read the template of input file and can return the new input with modified molecule
    """

    def __init__(
        self,
        link0=None,
        route=None,
        title=None,
        multiplicity=None,
        charge=None,
        additionals=None,
        atoms=None,
    ):
        self._link0 = link0 if link0 is not None else GaussianLink0()
        self._input = {
            "link0": asdict(self._link0),
            "route": route if route else "",
            "title": title,
            "multiplicity": multiplicity,
            "charge": charge,
            "atoms": atoms,
            "additionals": additionals if additionals else [],
        }

    @property
    def link0(self):
        return self._link0

    @property
    def route(self):
        return self._input["route"]

    @property
    def title(self):
        return self._input["title"]

    @property
    def multiplicity(self):
        return self._input["multiplicity"]

    @property
    def charge(self):
        return self._input["charge"]

    @property
    def atoms(self):
        return self._input["atoms"]

    @property
    def additionals(self):
        return self._input["additionals"]

    @staticmethod
    def read_inputf(input_file):
        if isinstance(input_file, str):
            input_file = Path(input_file)
        input_lines = iter(input_file.read_text().splitlines(keepends=True))
        # link0
        flag_link0 = False
        flag_route = False
        flag_title = False
        flag_mole = False
        flag_additionals = False

        _link0 = GaussianLink0()
        _route_line = "#"
        title = ""
        multiplicity = 1
        charge = 0
        atoms = []
        additionals = []

        line = next(input_lines)
        try:
            while line:  # loop to line without \n, which is the EOF
                # link0
                if flag_link0 and line.startswith(r"%"):
                    continue
                elif line.startswith(r"%"):
                    _link0.read_link0_commands(line)
                    line = next(input_lines)
                # route
                elif line.startswith(r"#"):
                    flag_link0 = True
                    _route_line += line[1:].strip()
                    line = next(input_lines)
                    flag_route = True
                # title
                elif flag_route and not flag_title:
                    line = next(input_lines)
                    title = line.strip()
                    flag_title = True
                # molecule
                elif flag_title and not flag_mole:
                    line = next(input_lines)
                    # spin and multiplicity
                    line = next(input_lines)
                    multiplicity = int(line.split()[1])
                    charge = int(line.split()[0])
                    # atom lines
                    line = next(input_lines)
                    while len(line.strip()) != 0:
                        atoms.append(line.split())
                        line = next(input_lines)
                    flag_mole = True
                # additionals, the end blank line has been read,
                # read all the lines remaining until any '--link1--'
                elif not flag_additionals:
                    line = next(input_lines)
                    if line.startswith("--link1--"):
                        break
                    else:
                        additionals.append(line.strip())
        except StopIteration:
            pass
        return GaussianInputBlock(
            **{
                "link0": _link0,
                "route": _route_line,
                "title": title,
                "multiplicity": multiplicity,
                "charge": charge,
                "atoms": atoms,
                "additionals": additionals,
            }
        )

    @staticmethod
    def _generate_mole_block(system):
        """generate molecule block from dpdata.System"""
        atom_names = system["atom_names"]
        atom_types = system["atom_types"]
        mole = ""
        for atom_type, positions in zip(
            [atom_names[atom_type] for atom_type in atom_types], system["coords"][0]
        ):
            mole += f"{atom_type} {positions[0]:f} {positions[1]:f} {positions[2]:f}\n"
        return mole

    def _generate_input_block(
        self,
        *,
        system: "dpdata.System",
        title=None,
        link0: GaussianLink0 = None,
        route=None,
        multiplicity=None,
        charge=None,
        additional_lines=None,
    ):
        """generate input block from dpdata.System"""
        from dpdata.periodic_table import Element

        atom_names = system["atom_names"]
        atom_types = system["atom_types"]
        # link0
        old_link0 = self._input["link0"]
        if link0 is not None:
            old_link0.update(asdict(link0))
        link0_block = GaussianLink0(**old_link0).generate_link0_commands()
        # title
        title = self._input["title"] if title is None else title
        if title is not None:
            title = title
        else:
            title = system.short_formula
        # route
        route = self._input["route"] if route is None else route
        if route.startswith("#"):
            route = route
        else:
            route = f"# {route}"
        # charge & multiplicity
        multiplicity = (
            self._input["multiplicity"] if multiplicity is None else multiplicity
        )
        charge = self._input["charge"] if charge is None else charge
        if charge is None:
            charge = 0
        if multiplicity is None:
            multiplicity = (
                sum([Element(atom_names[atom_type]).Z for atom_type in atom_types])
                - charge
            ) % 2 + 1
        # check multiplicity and charge
        if (
            sum([Element(atom_names[atom_type]).Z for atom_type in atom_types]) - charge
        ) % 2 != 0 and multiplicity % 2 == 1:
            raise ValueError(
                f"""For open-shell system, multiplicity must be even, got {
                    multiplicity}."""
            )
        elif (
            sum([Element(atom_names[atom_type]).Z for atom_type in atom_types]) - charge
        ) % 2 == 0 and multiplicity % 2 != 1:
            raise ValueError(
                f"""For closed-shell system, multiplicity must be odd, got {
                    multiplicity}."""
            )
        # mole
        mole_block = self._generate_mole_block(system)
        # additionals
        additionals = (
            self._input["additionals"] if additional_lines is None else additional_lines
        )
        if additionals is None:
            additionals = []

        return "\n".join(
            [
                link0_block,
                route,
                "",
                title,
                "",
                f"{charge} {multiplicity}",
                mole_block,
                "\n".join(additionals),
            ]
        )

    def update_input(
        self,
        *,
        system,
        title=None,
        link0: GaussianLink0 = None,
        route=None,
        multiplicity=None,
        charge=None,
        additional_lines=None,
    ):
        if additional_lines is None:
            additional_lines = []
        self._input = {
            "link0": asdict(link0) if link0 is not None else None,
            "route": route,
            "title": title,
            "multiplicity": multiplicity,
            "charge": charge,
            "atoms": self._generate_mole_block(system),
            "additionals": (
                additional_lines if len(additional_lines) else self.additionals
            ),
        }

    def write_input(
        self,
        input_path,
        *,
        system,
        title=None,
        link0: GaussianLink0 = None,
        route=None,
        multiplicity=None,
        charge=None,
        additional_lines=None,
    ):
        """write input file from dpdata.System"""
        result_input = self._generate_input_block(
            system=system,
            title=title,
            link0=link0,
            route=route,
            multiplicity=multiplicity,
            charge=charge,
            additional_lines=additional_lines,
        )
        Path(input_path).write_text(result_input)


def generate_gaussian_input(
    template_path=None,
    *,
    job_name,
    calculation_system,
    charge: int = None,
    multiplicity: int = None,
    route: Optional[str] = None,
    cpu: Optional[str] = None,
    nproc: Optional[str] = None,
    chk: Optional[str] = None,
    mem: Optional[str] = None,
    additionals: Optional[List[str]] = None,
):
    if additionals is None:
        additionals = []
    gaussian_input = GaussianInputBlock.read_inputf(template_path)
    if chk:
        gaussian_input.link0.chk = chk
    if cpu:
        gaussian_input.link0.cpu = cpu
    if mem:
        gaussian_input.link0.mem = mem
    if nproc:
        gaussian_input.link0.nproc = nproc
    if len(additionals) > 0:
        gaussian_input.additionals = additionals

    gaussian_input.update_input(
        system=calculation_system,
        title=job_name,
        link0=gaussian_input.link0,
        route=gaussian_input.route if route is None else route,
        multiplicity=multiplicity,
        charge=charge,
        additional_lines=additionals,
    )
    return gaussian_input


@OP.function
def prepGaussianCalculation(
    calculation_system: BigParameter("dpdata.System"),
    input_obj: BigParameter(GaussianInputBlock),
) -> {"input_dir": Artifact(List[Path]), "task_num": int, "task_name_list": List[str]}:
    result_dir = []
    task_num = len(calculation_system)
    task_name_list = []
    for work_idx, calculation_sys in enumerate(calculation_system):
        job_name = "gaussian_task_{:0{width}}".format(
            work_idx,
            width=len(str(task_num)) + 1,
        )
        work_dir = Path(job_name)
        with set_directory(work_dir, mkdir=True):
            input_obj.write_input(
                input_path="{}.gjf".format(job_name),
                system=calculation_sys,
                title=job_name,
                link0=input_obj.link0,
                route=input_obj.route,
                multiplicity=input_obj.multiplicity,
                charge=input_obj.charge,
                additional_lines=input_obj.additionals,
            )
        result_dir.append(work_dir)
        task_name_list.append(job_name)
    return {
        "input_dir": result_dir,
        "task_num": task_num,
        "task_name_list": task_name_list,
    }


@OP.function
def runGaussianCalculation(
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
    command = " ".join([command, "<", f"{job_name}.gjf", "| tee log"])
    if time_benchmark:
        command = f"""echo START: $(date "+%Y-%m-%d %H:%M:%S")>>\
            {time_log_path.as_posix()} && {command} && echo END: \
            $(date "+%Y-%m-%d %H:%M:%S")>>{time_log_path.as_posix()}"""
    ret, out, err = dflow_run_command(
        command,
        try_bash=True,
        shell=True,
        interactive=False,
        raise_error=False,
    )
    return {
        "output_dir": Path(input_dir),
        "log": Path("log"),
        "time_log_path": Path(time_log_path),
    }
