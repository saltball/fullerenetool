from typing import Any, Dict

import torch
from ase.calculators.calculator import Calculator, all_changes

from fullerenetool.constant import covalent_bond


class BondTopoBuilderCalculator(Calculator):
    name = "bond"
    implemented_properties = ["energy", "free_energy", "forces", "energies"]
    covalent_bond = covalent_bond
    calculator_parameters: Dict[str, Any] = dict(
        verbose=False,
        a0=1,
    )
    default_parameters = dict(calculator_parameters)

    def __init__(
        self,
        **kwargs,
    ):
        """
        params:
            kwargs:
                topo: nx.adjacency_matrix(nx.Graph).todense() of the molecule
                device: 'cpu' or 'cuda' for torch.device()

        """
        super().__init__(**kwargs)
        self.topo = kwargs["topo"]
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if kwargs.get("device", None) is None
            else torch.device(kwargs["device"])
        )

    def calculate(self, atoms=None, properties=None, system_changes=None):
        if properties is None:
            properties = self.implemented_properties
        if system_changes is None:
            system_changes = all_changes
        Calculator.calculate(self, atoms, properties, system_changes)
        self.run()

    def run(self):
        torch.autograd.set_detect_anomaly(True)
        device = self.device
        atoms = self.atoms
        precision = torch.float32
        a0 = torch.tensor(self.parameters["a0"], dtype=precision, device=device)
        covalent_bond = torch.tensor(self.covalent_bond, dtype=precision, device=device)

        # 获取原子编号和距离矩阵
        elements = torch.tensor(
            atoms.get_atomic_numbers(), dtype=torch.long, device=device
        )
        positions = torch.tensor(
            atoms.get_positions(), requires_grad=True, dtype=precision, device=device
        )
        distance_mat = (positions[:, None, :] - positions[None, :, :]).norm(dim=-1)

        # 获取连接矩阵
        connect_mat = self.topo
        connect_mat_rows, connect_mat_cols = connect_mat.nonzero()
        connect_mat_rows = torch.tensor(
            connect_mat_rows, dtype=torch.long, device=device
        )
        connect_mat_cols = torch.tensor(
            connect_mat_cols, dtype=torch.long, device=device
        )
        connect_mask = torch.tensor(connect_mat, dtype=torch.long, device=device)

        # 构建共价键长参数矩阵
        convalent_bond_mat = torch.zeros_like(distance_mat)
        convalent_bond_mat[connect_mat_rows, connect_mat_cols] = covalent_bond[
            elements[connect_mat_rows], elements[connect_mat_cols]
        ]

        # 构建键距矩阵
        bond_distance_mat = torch.zeros_like(distance_mat)
        bond_distance_mat[connect_mat_rows, connect_mat_cols] = distance_mat[
            connect_mat_rows, connect_mat_cols
        ]
        # 计算 d_dis
        d_dis = bond_distance_mat - convalent_bond_mat

        # 计算键能
        energy = (a0 * d_dis.pow(2)).sum()
        # 计算排斥能
        energy += ((0.01 / (distance_mat + 0.01) ** 2) * (1 - connect_mask)).sum()

        energy.backward()
        forces = -positions.grad

        self.results = {
            "energy": energy.detach().cpu().numpy(),
            "forces": forces.detach().cpu().numpy(),
        }
