import time
from pathlib import Path

from ase.io.extxyz import write_extxyz  # 假设这是写入 extxyz 文件的函数
from ase.optimize.lbfgs import LBFGS
from dflow.python import OP, OPIO, Artifact, BigParameter, OPIOSign


class DPSingletonManager:
    _instances = {}

    @classmethod
    def get_dp_calculator(cls, model_path):
        if model_path not in cls._instances:
            from deepmd.calculator import DP

            print(f"Loading DP model from {model_path}...")
            cls._instances[model_path] = DP(model_path)
        return cls._instances[model_path]

    @classmethod
    def clear_instance(cls, model_path=None):
        """Optional: Clear the instance for a specific model or all models."""
        if model_path:
            if model_path in cls._instances:
                del cls._instances[model_path]
        else:
            cls._instances.clear()


class DPCalculateEnergy(OP):
    def __init__(self):
        pass

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "atoms_file": Artifact(Path),
                "optimize": bool,
                "model_file": Artifact(Path),
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

        atoms = list(read_extxyz(Path(op_in["atoms_file"]).open("r"), -1))[-1]
        optimize = op_in["optimize"]
        output_name = Path(op_in["atoms_file"]).stem + ".xyz"

        # atoms_string = io.StringIO()
        st_time = time.perf_counter()
        atoms.calc = DPSingletonManager.get_dp_calculator(op_in["model_file"])
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
