from typing import Dict

import numpy as np
import torch
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from deepmd.pt.model.model import get_model
from deepmd.pt.train.wrapper import ModelWrapper
from deepmd.pt.utils.env import DEVICE, GLOBAL_PT_FLOAT_PRECISION
from deepmd.pt.utils.nlist import extend_input_and_build_neighbor_list
from dpdata.plugins.ase import ASEStructureFormat
from tqdm import tqdm


class EmbeddingCalculator:
    """Calculate the embedding of a given ase.Atoms object.

    Args:
        model_path: str
        device: str
        batch_size: int
        model_config: Dict

    """

    def __init__(
        self,
        model_path: str,
        device: str = None,
        batch_size: int = 1,
        model_config: Dict = None,
    ):
        self.model_path = model_path
        self.device = torch.device(device) if device is not None else DEVICE
        self.batch_size = batch_size
        self.model_config = model_config or self._default_config()

        # 初始化模型
        self.model = self._load_model()
        self._init_layers()

    def _default_config(self) -> Dict:
        """Return the default configuration for the embedding calculator.

        the default configuration is from dpa2 model
        """
        from ase.atom import chemical_symbols

        return {
            "type_map": chemical_symbols[1:],
            "descriptor": {
                "type": "dpa2",
                "repinit": {
                    "tebd_dim": 8,
                    "rcut": 6.0,
                    "rcut_smth": 0.5,
                    "nsel": 120,
                    "neuron": [25, 50, 100],
                    "axis_neuron": 12,
                    "activation_function": "tanh",
                    "three_body_sel": 40,
                    "three_body_rcut": 4.0,
                    "three_body_rcut_smth": 3.5,
                    "use_three_body": True,
                },
                "repformer": {
                    "rcut": 4.0,
                    "rcut_smth": 3.5,
                    "nsel": 40,
                    "nlayers": 6,
                    "g1_dim": 128,
                    "g2_dim": 32,
                    "attn2_hidden": 32,
                    "attn2_nhead": 4,
                    "attn1_hidden": 128,
                    "attn1_nhead": 4,
                    "axis_neuron": 4,
                    "update_h2": False,
                    "update_g1_has_conv": True,
                    "update_g1_has_grrg": True,
                    "update_g1_has_drrd": True,
                    "update_g1_has_attn": False,
                    "update_g2_has_g1g1": False,
                    "update_g2_has_attn": True,
                    "update_style": "res_residual",
                    "update_residual": 0.01,
                    "update_residual_init": "norm",
                    "attn2_has_gate": True,
                    "use_sqrt_nnei": True,
                    "g1_out_conv": True,
                    "g1_out_mlp": True,
                },
                "add_tebd_to_repinit_out": False,
            },
            "fitting_net": {
                "neuron": [240, 240, 240],
                "resnet_dt": True,
                "seed": 1,
                "_comment": " that's all",
            },
            "_comment": " that's all",
        }

    def _load_model(self) -> ModelWrapper:
        """Load the model from the given path."""
        state_dict = torch.load(self.model_path, map_location=self.device)
        model = get_model(self.model_config)
        model = ModelWrapper(model)

        # 修正state_dict键名
        # TODO: any general solution?
        new_state_dict = model.state_dict()
        for k, v in state_dict["model"].items():
            if "MP_traj_v024_alldata_mixu" in k:
                new_key = k.replace("MP_traj_v024_alldata_mixu", "Default")
                new_state_dict[new_key] = v
        model.load_state_dict(new_state_dict)
        model.to(self.device)
        return model

    def _init_layers(self):
        """The filter layers of the model, which generate the embedding."""
        self.filter_layers = []
        fitting_net = self.model.model.Default.get_fitting_net()
        for layer in fitting_net.filter_layers.networks[0].layers[:-1]:
            self.filter_layers.append(layer.to(self.device))

    def compute_embeddings(self, atoms: Atoms) -> Dict[str, np.ndarray]:
        """计算嵌入向量"""
        if isinstance(atoms, Atoms):
            atoms = atoms
        elif isinstance(atoms, list):
            atoms = Atoms([atom for atom in atoms])
        else:
            raise ValueError(
                "Invalid input type. Input must be Atoms or list of Atoms."
            )
        atomic_emb = self._compute_single(atoms)
        return atomic_emb

    def _compute_single(self, atoms: Atoms) -> np.ndarray:
        """计算单个结构的嵌入"""
        # 准备输入数据
        frame_data = self._preprocess_input(atoms)
        inputs = self._prepare_tensors(frame_data)

        # 模型前向传播
        with torch.no_grad():
            descriptor_out = self.model.model.Default.get_descriptor()(**inputs)
            xx = descriptor_out[0]
            for layer in self.filter_layers:
                xx = layer(xx)

        return xx.detach().cpu().numpy()

    def _preprocess_input(self, atoms: Atoms) -> Dict:
        """数据预处理"""
        atoms.calc = SinglePointCalculator(
            atoms,
            # energy=atoms.get_potential_energy(),
            # forces=atoms.get_array('force'),
        )
        return ASEStructureFormat().from_system(atoms)

    def _prepare_tensors(self, frame_data: Dict) -> Dict:
        """Prepare tensors for the model input."""
        coords = frame_data["coords"]
        atom_types = frame_data["atom_types"]
        box = frame_data["cells"]
        natoms = len(atom_types)
        nframes = coords.shape[0]
        coord_input = np.array(
            coords.reshape([-1, natoms, 3]),
        )
        box_input = np.array(
            box.reshape([-1, 3, 3]),
        )
        atom_types = np.tile(atom_types, nframes).reshape(nframes, -1)
        atom_types_input = np.array(atom_types.reshape([-1, natoms]), dtype=np.int32)
        coord_input = torch.tensor(
            coord_input,
            dtype=GLOBAL_PT_FLOAT_PRECISION,
            device=DEVICE,
        )
        box_input = torch.tensor(
            box_input,
            dtype=GLOBAL_PT_FLOAT_PRECISION,
            device=DEVICE,
        )
        atom_types_input = torch.tensor(
            atom_types_input, dtype=torch.long, device=DEVICE
        )

        # Deal with the input type cast
        # see https://github.com/deepmodeling/deepmd-kit/blob/r3.0/deepmd/pt/infer/deep_eval.py # noqa: E501
        cc, bb, fp, ap, input_prec = self.model.model.Default.input_type_cast(
            coord_input, box=box_input
        )
        (
            extended_coord,
            extended_atype,
            mapping,
            nlist,
        ) = extend_input_and_build_neighbor_list(
            cc,
            atom_types_input,
            self.model.model.Default.get_rcut(),
            self.model.model.Default.get_sel(),
            mixed_types=self.model.model.Default.mixed_types(),
            box=bb,
        )

        return {
            "extended_coord": extended_coord,
            "extended_atype": extended_atype,
            "nlist": nlist,
            "mapping": mapping,
        }


class EmbeddingManager:
    """嵌入计算器管理类（单例模式）"""

    _instances = {}

    @classmethod
    def get_calculator(cls, model_path: str, **kwargs) -> EmbeddingCalculator:
        """获取计算器实例"""
        if model_path not in cls._instances:
            cls._instances[model_path] = EmbeddingCalculator(
                model_path=model_path, **kwargs
            )
        return cls._instances[model_path]

    @classmethod
    def clear_instance(cls, model_path: str = None):
        """清理实例"""
        if model_path:
            if model_path in cls._instances:
                del cls._instances[model_path]
        else:
            cls._instances.clear()


if __name__ == "__main__":
    import os
    from pathlib import Path

    os.chdir(Path(__file__).parent)
    MODEL_PATH = "../DPA2_medium_28_10M_rc0.pt"

    calculator = EmbeddingManager.get_calculator(
        model_path=MODEL_PATH,
        device=DEVICE,
    )

    from ase.io import read

    atoms_list = read("...xyz", index=":")

    for atoms in tqdm(atoms_list):
        embeddings = calculator.compute_embeddings(atoms)
        # print(embeddings)
