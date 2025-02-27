import logging
import warnings
from functools import cached_property
from typing import Iterable, List, Optional

import ase
import networkx as nx
import numpy as np

from fullerenetool.constant import covalent_bond
from fullerenetool.fullerene._base_fullerene import BaseAbstartFullerene
from fullerenetool.fullerene.cage import (
    DEFAULT_MAX_STEPS,
    BaseAbstactGraph,
    CageGraph,
    FullereneCage,
)
from fullerenetool.logger import logger
from fullerenetool.operator.calculator.bond_topo_builder import (
    BondTopoBuilderCalculator,
)
from fullerenetool.operator.checker.molecule import filter_ghost_atom
from fullerenetool.operator.fullerene.add_exo import add_out_of_cage, get_cage_addon_pos
from fullerenetool.operator.graph import get_graph_from_atoms


class BaseDerivativeFullerene(BaseAbstartFullerene):
    def __init__(self, atoms, cage: FullereneCage):
        self._atoms = atoms
        self._cage = cage
        self.derivatives = self._get_derivatives()
        self.cage_graph = self._cage.graph
        self.cage_atom_index = []
        for atomidx, atom in enumerate(self._atoms):
            if np.any(
                np.all(
                    np.isclose(
                        atom.position,
                        self._cage.atoms.positions,
                    ),
                    axis=1,
                )
            ):
                self.cage_atom_index.append(atomidx)
        self.derivative_atom_index = [
            index
            for index in range(len(self._atoms))
            if index not in self.cage_atom_index
        ]
        if len(self.derivatives) == 0:
            logging.warning("No derivatives found in the fullerene")

    def _get_derivatives(self):
        """Return the atoms not on the cage, which is considered as derivative atoms"""
        derivatives = []
        for atom in self._atoms:
            if atom not in self._cage.atoms:
                derivatives.append(atom)
        return ase.Atoms(derivatives)

    def get_total_graph(self):
        return get_graph_from_atoms(self._atoms, only_top3=False)

    def get_derivative_graph(self):
        return get_graph_from_atoms(self.derivatives, only_top3=False)

    def get_first_derivative_graph(self):
        """Get the graph with atoms in the first neighboring of the derivative atoms

        return type: networkx.Graph
        """

    @property
    def cage(self) -> "FullereneCage":
        return self._cage

    @property
    def cages(self) -> List["FullereneCage"]:
        return [self._cage]

    @property
    def graph(self) -> CageGraph:
        return self.get_total_graph()

    def _check_atoms(self, atoms):
        pass

    def __str__(self):
        return "BaseDerivativeFullerene({}):Cage({}),Dev({})".format(
            self.atoms.get_chemical_formula(),
            self._cage.atoms.get_chemical_formula(),
            self.derivatives.get_chemical_formula(),
        )

    @classmethod
    def from_cage(
        cls,
        cage: FullereneCage,
        addons: ase.Atoms,
        addon_to: int,
        addon_bond_length=None,
    ):
        """Generate a derivative fullerene from a cage and addons

        Args:
            cage (FullereneCage): The cage.
            addons (ase.Atoms): The addons.
            addon_to (int): The index of the atom to add the addons.
            addon_bond_length (float, optional):
                The bond length between the addon and the atom. Defaults to None.

        Returns:
            cls: The derivative fullerene.
        """
        addons_pos, addon_vec = get_cage_addon_pos(
            cage.atoms,
            addon_to,
            addons,
            bond_length=addon_bond_length,
            return_vec=True,
        )

        return cls(
            atoms=add_out_of_cage(
                cage.atoms,
                addons,
                addons_pos=addons_pos,
                addons_vec=addon_vec,
            ),
            cage=cage,
        )


class DerivativeGroup:
    def __init__(
        self,
        atoms: ase.Atoms,
        graph: nx.Graph,
        bond_length: float = None,
        addon_atom_idx: Optional[int] = None,
    ):
        adj_matrix = nx.adjacency_matrix(graph).todense()
        assert adj_matrix.shape[0] == len(atoms)
        if addon_atom_idx is not None:  # addon_atom_idx offered, addon site is signed
            if atoms[addon_atom_idx].symbol != "X":
                raise ValueError(
                    "The addon atom is not a pseudo atom, please specify the pseudo."
                )
        else:
            if "X" in atoms.get_chemical_symbols():
                addon_atom_idx = np.where(
                    np.array(atoms.get_chemical_symbols()) == "X"
                )[0][0]
            else:
                warnings.warn(
                    "No addon atom is signed, use the first atom as the addon atom."
                )
                addon_atom_idx = 0
                first_neighbor = adj_matrix[0].nonzero()[0][0]
                bond_vec = atoms.positions[first_neighbor] - atoms.positions[0]
                pesudo_pos = (
                    -bond_vec / np.linalg.norm(bond_vec) * bond_length
                    + atoms.positions[0]
                )
                atoms = ase.Atoms(
                    "X" + atoms.symbols,
                    [pesudo_pos, *atoms.positions],
                )
                new_adj_matrix = np.zeros(
                    [adj_matrix.shape[0] + 1, adj_matrix.shape[1] + 1]
                )
                new_adj_matrix[1:, 1:] = adj_matrix
                new_adj_matrix[0, 1] = 1
                new_adj_matrix[1, 0] = 1
                adj_matrix = new_adj_matrix
                graph = nx.from_numpy_array(adj_matrix)

        if atoms[addon_atom_idx].symbol == "X":
            if adj_matrix[addon_atom_idx].nonzero()[0].sum() != 1:
                raise ValueError(
                    "The graph of the derivative group is not valid, "
                    "addon site is connected to {} atoms.".format(
                        adj_matrix[addon_atom_idx].nonzero()
                    )
                )
            self.first_neighbor = adj_matrix[addon_atom_idx].nonzero()[0][0]
            add_vec = (
                atoms.positions[self.first_neighbor] - atoms.positions[addon_atom_idx]
            )
            self.add_vec = add_vec / np.linalg.norm(add_vec)
        else:
            raise RuntimeError(
                "This case is not implemented yet that \
                    the addon atom is not a pseudo atom."
            )
        self.addon_atom_idx = addon_atom_idx
        self.atoms = atoms
        self.graph = graph

    def addto(
        self, idx, fullerene: BaseAbstartFullerene, outside=True, check=True
    ) -> BaseDerivativeFullerene:
        # check whether the idx of the fullerene is a valid addon point
        if idx not in fullerene.graph.nodes:
            raise ValueError("The index {} is not a valid addon point.".format(idx))
        if check:
            if nx.adjacency_matrix(fullerene.graph).todense()[idx].sum() > 3:
                raise ValueError(
                    "The addon point {} is not a valid addon point, \
                        which is connected to {} atoms.".format(
                        idx, nx.adjacency_matrix(fullerene.graph).todense()[idx].sum()
                    )
                )
        dev_pos, dev_vec = get_cage_addon_pos(
            fullerene.atoms,
            idx,
            self.atoms,
            self.first_neighbor,
            None,
        )
        dev = add_out_of_cage(
            fullerene.atoms,
            filter_ghost_atom(self.atoms),
            addons_pos=dev_pos,
            addons_vec=dev_vec,
            addons_index=self.addon_atom_idx,
            shake_num=50,
            check=True,
        )
        dev = ase.Atoms(dev[dev.get_atomic_numbers() != 0])
        return BaseDerivativeFullerene(
            atoms=dev,
            cage=fullerene.cages[0],
        )

    @property
    def name(self):
        return "{}".format(filter_ghost_atom(self.atoms).get_chemical_formula())

    def __str__(self):
        return "DerivativeGroup({})".format(self.name)


class DerivativeFullereneGraph(BaseAbstactGraph):

    def __init__(
        self,
        adjacency_matrix: np.ndarray,
        cage_elements: Iterable[str],
        addons: Iterable[DerivativeGroup],
        **kwargs,
    ):
        if adjacency_matrix is None:
            raise ValueError("The adjacency_matrix is required.")
        self.adjacency_matrix = adjacency_matrix
        addons_elements = [
            filter_ghost_atom(addon.atoms).get_chemical_symbols() for addon in addons
        ]
        addons_elements = [
            item
            for sublist in addons_elements
            for item in (sublist if isinstance(sublist, list) else [sublist])
        ]  # flatten
        self.node_elements = [*cage_elements, *addons_elements]
        self._check_node_elements()
        self.kwargs = kwargs
        assert (
            len(cage_elements) + len(addons_elements) == adjacency_matrix.shape[0]
        ), "The number of elements must be equal to the number of nodes,\
            got {} and {}+{}".format(
            adjacency_matrix.shape[0],
            len(cage_elements),
            len(addons_elements),
        )
        self.cage_elements = cage_elements
        self.addons_elements = addons_elements
        self.addons = addons
        self.cage_atom_index = range(len(cage_elements))
        self.addon_atom_index = []
        addon_idx_count = len(self.cage_elements)
        for addon in addons:
            self.addon_atom_index.append(addon.addon_atom_idx + addon_idx_count)
            addon_idx_count += len(filter_ghost_atom(addon.atoms))

    @property
    def addon_sites_candidate(self):
        coord_nums = self.adjacency_matrix[self.cage_atom_index].sum(axis=1)
        return np.where(coord_nums == 3)[0]

    def __str__(self):
        return "DerivativeFullereneGraph object with\
            {} nodes and {} edges, node_zs: {}".format(
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
            self.node_elements,
        )

    @cached_property
    def _circle_finder(self):
        from fullerenetool.algorithm import dual_graph

        return dual_graph.py_graph_circle_finder(
            int(len(self.graph.edges)), np.array(self.graph.edges).astype(int).data
        )

    @property
    def cage_graph(self):
        return CageGraph(
            self.adjacency_matrix[: len(self.cage_elements), : len(self.cage_elements)],
            node_elements=self.cage_elements,
            atoms=self.kwargs.get("atoms"),
        )

    def generate_atoms_with_addons(
        self,
        algorithm="cagethenaddons",
        init_pos=None,
        traj=None,
        max_steps=DEFAULT_MAX_STEPS,
        b2n2_steps=50,
        use_gpu=False,
        add_outside=True,
        check=True,
    ) -> ase.Atoms:
        if init_pos is not None and len(init_pos) < len(self.node_elements):
            # the initial positions is not compelete provided for the addons
            logger.info(
                "The initial positions is less than the number of nodes:"
                "{} < {}".format(len(init_pos), len(self.node_elements))
                + ", so some atoms will be placed with optimal positions."
            )
            new_init_pos = np.zeros([len(self.node_elements), 3])
            new_init_pos[: len(init_pos), :] = init_pos
            if add_outside:
                inside_center = init_pos[: len(self.cage_elements)].mean(0)
                for add_pos_index in self.addon_atom_index:
                    addon_index = self.adjacency_matrix[add_pos_index].nonzero()[0]
                    if addon_index[0].size != 1:
                        raise ValueError(
                            "Cannot add the addon because the connection is not 1, "
                            + "got {}".format(addon_index[0].size)
                        )
                    addon_pos = init_pos[addon_index[0]]
                    new_init_pos[add_pos_index] = (
                        inside_center
                        + (addon_pos - inside_center) * 1.2
                        + covalent_bond[
                            int(self.node_elements[add_pos_index]),
                            int(self.node_elements[addon_index[0]]),
                        ]
                        * ((addon_pos - inside_center))
                        / np.linalg.norm((addon_pos - inside_center))
                    )

            else:
                raise NotImplementedError(
                    "Add indise not implemented yet, "
                    "please provide the positions for all atoms."
                )
        elif init_pos is not None and len(init_pos) > len(self.node_elements):
            warnings.warn(
                "The initial positions is longer than the number of nodes:"
                "{} > {}".format(len(init_pos), len(self.node_elements))
            )
            new_init_pos = init_pos[: len(self.cage_elements)]
        else:  # None or complete
            if init_pos is None:
                init_pos = self.get("atoms").positions if self.get("atoms") else None
            new_init_pos = init_pos
        if algorithm == "cagethenaddons":
            # cage
            cage_ = self.cage_graph.generate_atoms(
                elements=self.cage_elements,
                algorithm="b2n2thenb2",
                init_pos=(
                    new_init_pos[: len(self.cage_elements), :]
                    if new_init_pos is not None
                    else None
                ),
                traj="cage_" + traj if traj else None,
                b2n2_steps=b2n2_steps,
                max_steps=max_steps,
                use_gpu=use_gpu,
            )
            # add addons
            cage = FullereneCage(atoms=cage_)
            derivated_atoms = BaseDerivativeFullerene(
                atoms=cage.atoms,
                cage=cage,
            )
            for derivation in self.addons:
                derivated_atoms = derivation.addto(
                    self.adjacency_matrix[
                        len(derivated_atoms.atoms) + derivation.addon_atom_idx
                    ].nonzero()[0][0],
                    derivated_atoms,
                    check=check,
                )
            from ase.optimize.lbfgs import LBFGS

            calc = BondTopoBuilderCalculator(
                topo=nx.adjacency_matrix(self.graph).todense(),
                device="cuda" if use_gpu else "cpu",
            )
            atoms = ase.Atoms(
                symbols=self.node_elements,
                positions=derivated_atoms.positions,
                calculator=calc,
            )
            opt = LBFGS(
                atoms,
                trajectory="addons_" + traj if traj else None,
            )
            opt.run(fmax=0.001, steps=max_steps)
            return atoms
        else:
            raise NotImplementedError("Algorithm {} not implemented".format(algorithm))

    def visualize(self, ax=None, color="jmol", atoms=None, **kwargs):

        from ase.data.colors import cpk_colors, jmol_colors
        from matplotlib import pyplot as plt

        from fullerenetool.fullerene.visualize.cage import planarity_graph_pos

        cage_pos, project_func = planarity_graph_pos(
            FullereneCage(atoms[self.cage_atom_index]),
            return_project_matrix=True,
        )
        pos = {idx: cage_pos[idx][:2] for idx in range(len(cage_pos))}

        addons_pos = atoms[
            [i for i in range(len(atoms)) if i not in self.cage_atom_index]
        ].positions

        projected_addons_pos = project_func(addons_pos)

        all_pos = {
            **pos,
            **{
                idx + len(pos): projected_addons_pos[idx][:2]
                for idx in range(len(projected_addons_pos))
            },
        }

        if color == "jmol":
            coloring = jmol_colors
        elif color == "cpk":
            coloring = cpk_colors
        else:
            coloring = color
        colors = [
            coloring[node_z]
            for node_z in list(nx.get_node_attributes(self.graph, "z").values())
        ]

        title = "{} Cage Graph with Coloring in ase format".format(
            atoms.get_chemical_formula()
        )

        if ax is None:
            nx.draw(
                self.cage_graph.graph,
                pos,
                with_labels=True,
                node_color=[
                    (
                        "b"
                        if i not in self.addon_sites_candidate
                        else colors[: len(pos)][i]
                    )
                    for i in range(len(pos))
                ],
                node_size=250,
            )
            nx.draw(
                self.graph.subgraph(
                    list(
                        range(
                            self.cage_graph.graph.number_of_nodes(),
                            self.graph.number_of_nodes(),
                        )
                    )
                ),
                {
                    i: all_pos[i]
                    for i in range(len(all_pos))
                    if i >= self.cage_graph.graph.number_of_nodes()
                },
                with_labels=False,
                node_color=colors[len(self.cage_elements) :],
                alpha=0.5,
            )
            plt.title(title)
            plt.tight_layout()
            plt.show()
        else:
            nx.draw(
                self.cage_graph.graph,
                pos,
                with_labels=True,
                node_color=colors,
                node_size=500,
                ax=ax,
            )
            nx.draw(
                self.graph.subgraph(
                    list(
                        range(
                            self.cage_graph.graph.number_of_nodes(),
                            self.graph.number_of_nodes(),
                        )
                    )
                ),
                {
                    i: pos[i]
                    for i in pos
                    if i >= self.cage_graph.graph.number_of_nodes()
                },
                with_labels=True,
                node_color=coloring[len(self.cage_elements) :],
                node_size=100,
                ax=ax,
            )


def addons_to_adj_mat(new_adj, addon_idx, cage_idx):
    new_adj[addon_idx, cage_idx] = 1
    new_adj[cage_idx, addon_idx] = 1


def addons_to_fullerene(
    addons_list,
    addon_site_idx_list,
    cage: FullereneCage,
    cage_graph_adj: np.ndarray,
    addon_bond_length=None,
    max_steps=500,
    traj=None,
):
    assert len(addons_list) == len(
        addon_site_idx_list
    ), "The number of addons must be equal to the number of addon_site_idx"
    ADD_NUM = sum([len(addon.graph.nodes) - 1 for addon in addons_list])
    origin_len = len(cage_graph_adj)
    new_adj = np.zeros((origin_len + ADD_NUM, origin_len + ADD_NUM))
    for i in range(origin_len):
        for j in range(origin_len):
            new_adj[i, j] = cage_graph_adj[i, j]

    add_flag = origin_len
    for idx, addto in enumerate(addon_site_idx_list):
        addon_index = addons_list[idx].addon_atom_idx
        addons_to_adj_mat(new_adj, add_flag + addon_index, addto)
        for bond1, bond2 in addons_list[idx].graph.edges:
            if addon_index in [bond1, bond2]:
                continue
            if bond1 > addon_index:
                bond1 = bond1 - 1
            if bond2 > addon_index:
                bond2 = bond2 - 1
            addons_to_adj_mat(new_adj, add_flag + bond1, add_flag + bond2)
        add_flag = add_flag + len(addons_list[idx].graph.nodes) - 1

    dev_graph = DerivativeFullereneGraph(
        adjacency_matrix=new_adj,
        cage_elements=cage.graph.node_elements,
        addons=addons_list,
    )
    dev_mol = dev_graph.generate_atoms_with_addons(
        algorithm="cagethenaddons",
        init_pos=cage.atoms.positions,
        max_steps=max_steps,
        check=False,
        traj="cagethenaddons_" + traj if traj else None,
    )
    dev_graph = DerivativeFullereneGraph(
        adjacency_matrix=new_adj,
        cage_elements=cage.graph.node_elements,
        addons=addons_list,
        atoms=dev_mol,
    )
    return dev_graph, dev_mol
