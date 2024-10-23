import time
from pathlib import Path

from ase.io.extxyz import write_extxyz  # 假设这是写入 extxyz 文件的函数
from ase.optimize.lbfgs import LBFGS
from dflow.python import OP, OPIO, Artifact, BigParameter, OPIOSign


class CalculateEnergy(OP):
    def __init__(self):
        pass

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "atoms_file": Artifact(Path),
                "output_name": BigParameter(str),
                "optimize": bool,
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "energy": BigParameter(float),
                # "atoms_string": BigParameter(str),
                "atoms_xyz": Artifact(str),
                "time": BigParameter(float),
            }
        )

    @OP.exec_sign_check
    def execute(
        self,
        op_in: OPIO,
    ) -> OPIO:
        from ase.io.extxyz import read_extxyz

        from fullerenetool.logger import logger
        from fullerenetool.operator.calculator.mace_calculator import mace_mp

        atoms = list(read_extxyz(Path(op_in["atoms_file"]).open("r"), -1))[-1]
        optimize = op_in["optimize"]
        output_name = op_in["output_name"] + ".xyz"

        # atoms_string = io.StringIO()
        st_time = time.perf_counter()
        atoms.calc = mace_mp()
        if optimize:
            opt = LBFGS(atoms)
            opt.run(fmax=0.01)
        else:
            atoms.get_potential_energy()
        fl_time = time.perf_counter() - st_time
        write_extxyz(
            # atoms_string,
            Path(output_name).open("w"),
            atoms,
        )
        logger.info(f"run time: {fl_time:.6f} s")

        op_out = OPIO(
            {
                "energy": atoms.get_potential_energy(),
                # "atoms_string": atoms_string.getvalue(),
                "atoms_xyz": Path(output_name),
                "time": fl_time,
            }
        )

        return op_out
