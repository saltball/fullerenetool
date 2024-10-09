from abc import ABC, abstractmethod
from functools import cached_property
from typing import Iterable, List

import ase
import networkx as nx
import numpy as np
from ase.neighborlist import NeighborList, natural_cutoffs
from ase.optimize.optimize import DEFAULT_MAX_STEPS

from fullerenetool.fullerene import BaseAbstartFullerene
from fullerenetool.logger import logger
from fullerenetool.operator.calculator.bond_topo_builder import (
    BondTopoBuilderCalculator,
)
from fullerenetool.operator.graph import get_graph_from_atoms, nx_to_nauty


def _fullerene_cage_init_check(atoms):
    """Check the input atoms object is a fullerene cage.

    The `atoms` object must be an instance of `ase.Atoms`,
    and must contain carbon atoms only.

    Args:
        atoms: The atoms object to check.

    Raises:
        ValueError: If the atoms object is not an instance of `ase.Atoms`.
        ValueError: If the atoms object does not contain carbon atoms.
        ValueError: If the atoms object does not have at least one atom.

    Returns:
        ase.Atoms: The input atoms object.
    """
    if not isinstance(atoms, ase.Atoms):
        raise ValueError("The atoms object must be an instance of ase.Atoms.")
    if not atoms:
        raise ValueError("The atoms object must be provided")
    if not (atoms.get_atomic_numbers() == 6).all():
        raise ValueError(
            "The atoms object must be made up of only carbon atoms, "
            "got compound: {}".format(atoms.symbols)
        )
    if len(atoms) < 1:
        raise ValueError("The atoms object must have at least one atom.")
    return atoms


def check_carbon_cage(atoms: ase.Atoms):
    """Checks that the atoms object is a carbon cage.

    The `CageGraph` is expected to be a full-carbon cage with no
    restrictions on the number of atoms or the shape of the cage.

    Args:
        atoms (ase.Atoms): The atoms that make up the fullerene.
    """
    return _fullerene_cage_init_check(atoms)


class BaseAbstactGraph(ABC):

    @cached_property
    def graph(self) -> nx.Graph:
        """The networkx graph object of the fullerene graph."""
        graph = nx.from_numpy_array(
            self.adjacency_matrix,
            create_using=nx.Graph,
        )
        for node, z in zip(graph.nodes, self.node_elements):
            graph.nodes[node]["z"] = z
        return graph

    def generate_atoms(
        self,
        elements: List[str] = None,
        algorithm="b2n2",
        init_pos=None,
        traj=None,
        max_steps=DEFAULT_MAX_STEPS,
        use_gpu=False,
    ) -> ase.Atoms:
        """Generate atoms from the CageGraph.

        Args:
            elements (List[str]): The list of elements to be assigned to the atoms.
            algorithm (str): The algorithm to use to generate the atoms. Currently
                only "b2n2" is supported.
            init_pos (List[np.ndarray], optional): The initial positions of the atoms.
                Defaults to None. Useful for generating the new molecule based on the
                existing atoms. If not provided, random positions will be assigned.
            traj (str, optional): The path to save the trajectory of the atoms.
                Defaults to None, which means no trajectory will be saved.
            max_steps (int, optional): The maximum number of steps to run the
                optimization. Defaults to None, which means no limit, be caution to
                set a reasonable value, as it will consume a lot of time.
            use_gpu (bool, optional): Whether to use GPU for calculation.
                Defaults to False.

        Returns:
            ase.Atoms: The atoms structure of the CageGraph by certaining the
                elements.
        """
        # TODO(algorithm): implement generate atoms from CageGraph
        from ase import Atoms
        from ase.data import chemical_symbols
        from ase.optimize.lbfgs import LBFGS

        calc = BondTopoBuilderCalculator(
            topo=nx.adjacency_matrix(self.graph).todense(),
            device="cuda" if use_gpu else "cpu",
        )
        elements = elements or [
            chemical_symbols[node_z] for node_z in self.node_elements
        ]
        if self.kwargs.get("atoms") and init_pos is None:
            init_pos = self.kwargs["atoms"].positions
        atoms = Atoms(
            symbols=elements,
            positions=(
                [
                    np.random.uniform(low=-1, high=1, size=3)
                    for _ in range(len(elements))
                ]
                if init_pos is None
                else init_pos
            ),
            calculator=calc,
        )
        opt = LBFGS(atoms, trajectory=traj)
        opt.run(fmax=0.001, steps=max_steps)
        return atoms

    @abstractmethod
    def __str__(self):
        raise NotImplementedError

    @abstractmethod
    @cached_property
    def _circle_finder(self):
        raise NotImplementedError

    @property
    def nauty_graph(self):
        """Returns the nauty graph representation of the fullerene graph."""
        return nx_to_nauty(self.graph)

    def circles(self):
        """Calculates the circles in the graph.

        Returns:
            List[np.ndarray[int]]: The list of circles in the graph.
        """
        return self._circle_finder.get_face_vertex_list()

    def _get_circle_matrix(self):
        edge_list = self._circle_finder.get_dual_edge_list()
        circleADJ = np.zeros(
            [self._circle_finder.face_size, self._circle_finder.face_size]
        )
        for item in edge_list:
            circleADJ[item[0], item[1]] = 1
            circleADJ[item[1], item[0]] = 1
        return circleADJ

    def _init_check(self):
        # check the graph is connected all, not seperate
        if not nx.is_connected(self.graph):
            subgraphs = [
                len(c)
                for c in sorted(
                    nx.connected_components(self.graph), key=len, reverse=True
                )
            ]
            raise ValueError(
                "The graph must be connected, got {} graph componets: {}".format(
                    len(subgraphs), subgraphs
                )
            )
        if not nx.is_planar(self.graph):
            raise ValueError("The graph is not planar, cannot be as a Cage.")
        if len(self.node_elements) != len(self.adjacency_matrix):
            raise ValueError(
                "The number of node_zs must be equal to the number of atoms,"
                "got {} and {}".format(
                    len(self.node_elements), len(self.adjacency_matrix)
                )
            )

    def _check_node_elements(self):
        """Checks the node_elements are valid."""
        for nidx, node_element in enumerate(self.node_elements):
            if node_element not in ase.data.atomic_numbers.values():
                try:
                    node_ele = ase.data.atomic_numbers[node_element]
                except KeyError:
                    raise ValueError(
                        "Unknown element: {} in node_zs, must be one of {}".format(
                            node_element, list(ase.data.atomic_numbers.keys())
                        )
                    )
                self.node_elements[nidx] = node_ele


class CageGraph(BaseAbstactGraph):

    def __init__(
        self,
        adjacency_matrix: np.ndarray,
        node_elements: Iterable[str] = None,
        check=True,
        **kwargs,
    ):
        """Initializes a CageGraph object.

        The CageGraph object is a representation of a fullerene graph. It
        assumes that the fullerene is a cage with no restrictions on the
        shape of the cage.

        All atoms are connected to each other by bonds. The graph is
        represented by an adjacency matrix.

        Note the following for inputs:
        - The adjacency matrix is a square matrix that represents the bonds
          between atoms. The matrix is symmetric and the diagonal is zero.
        - The node_zs is a list of the elements of the nodes in the graph.
          Defaults to `6` for all nodes, means a carbon cage.

        Args:
            adjacency_matrix (np.ndarray): The adjacency matrix of the cage.
            node_zs (Iterable[str]): The elements of the nodes in the
              graph. Defaults to `6` for all nodes.
        """
        if adjacency_matrix is None:
            raise ValueError("The adjacency_matrix is required.")
        self.adjacency_matrix = adjacency_matrix
        self.node_elements = (
            node_elements
            if node_elements is not None
            else ([6] * len(adjacency_matrix))
        )
        self._check_node_elements()
        if check:
            self._init_check()
        self.circle_matrix = self._get_circle_matrix()
        self.kwargs = kwargs

    @cached_property
    def _circle_finder(self):
        from fullerenetool.algorithm import dual_graph

        circle_finder = dual_graph.py_graph_circle_finder(
            int(len(self.graph.edges)), np.array(self.graph.edges).astype(int).data
        )
        return circle_finder

    @classmethod
    def from_carbon_cage(cls, atoms: ase.Atoms):
        """Creates a CageGraph object from an Carbon only ase.Atoms object."""
        fullerene = check_carbon_cage(atoms)
        return cls.from_atoms(fullerene)

    @classmethod
    def from_atoms(cls, atoms: ase.Atoms):
        """Creates a CageGraph object from a ase.Atoms object."""
        cutoffs = natural_cutoffs(atoms)
        neighborList = NeighborList(cutoffs, self_interaction=False, bothways=True)
        neighborList.update(atoms)
        adjacency_matrix = neighborList.get_connectivity_matrix(sparse=False)
        if (adjacency_matrix.sum(-1) != 3).any():
            adjacency_matrix = np.zeros_like(adjacency_matrix)
            logger.warning(
                "More than 3 neighbors by ase found for input atoms"
                "Trying using first 3 shortest distance."
            )
            neighborDis: np.ndarray = atoms.get_all_distances()
            neighborSorted = np.argsort(neighborDis)
            adjacency_matrix[np.arange(len(atoms))[:, None], neighborSorted[:, 1:4]] = 1
        adjacency_matrix = np.array(adjacency_matrix)
        return CageGraph(
            adjacency_matrix=adjacency_matrix,
            node_zs=atoms.get_atomic_numbers(),
            atoms=atoms,
        )

    def __str__(self) -> str:
        return "CageGraph object with {} nodes and {} edges, node_zs: {}".format(
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
            self.node_elements,
        )

    def visualize(self, ax=None, color="jmol", **kwargs):
        """_summary_

        Args:
            ax (_type_, optional): _description_. Defaults to None.
            color (str, optional): _description_. Defaults to 'jmol'.
                see ase.data.colors for more details.
        """
        from ase.data.colors import cpk_colors, jmol_colors
        from matplotlib import pyplot as plt

        from fullerenetool.fullerene.visualize.cage import planarity_graph_pos

        atoms = self.generate_atoms()
        pos = planarity_graph_pos(
            FullereneCage(atoms),
        )
        pos = {idx: pos[idx][:2] for idx in range(len(pos))}

        if color == "jmol":
            coloring = jmol_colors
        elif color == "cpk":
            coloring = cpk_colors
        else:
            raise ValueError("Unknown color scheme: {}".format(color))
        colors = [
            coloring[node_z]
            for node_z in list(nx.get_node_attributes(self.graph, "z").values())
        ]

        title = "{} Cage Graph with Coloring in ase format".format(
            atoms.get_chemical_formula()
        )
        if ax is None:
            nx.draw(
                self.graph,
                pos,
                with_labels=True,
                node_color=colors,
                node_size=500,
            )
            plt.title(title)
            plt.tight_layout()
            plt.show()
        else:
            nx.draw(
                self.graph,
                pos,
                with_labels=True,
                node_color=colors,
                node_size=500,
                ax=ax,
            )
            ax.set_title(title)


class FullereneCage(BaseAbstartFullerene):
    """`FullereneCage` is a class that represents a fullerene cage.

    The cage means a large molecule that is formed by connecting atoms.
    It may contain carbon atoms and other atoms.
    """

    @property
    def graph(self, pick_atom_numbers=None) -> CageGraph:
        """Get the graph of the fullerene cage.

        Args:
            pick_atom_numbers (list, optional): The list of element numbers to be
                included in the graph. Defaults to None, which means all atoms.

        Returns:
            nx.Graph: The graph of the fullerene cage.
        """
        picked_atoms = self.atoms
        if pick_atom_numbers is not None:
            picked_atoms = ase.Atoms(
                [
                    self.atoms[self.atoms.get_atomic_numbers() == atom_number]
                    for atom_number in pick_atom_numbers
                ]
            )
        graph = get_graph_from_atoms(picked_atoms, only_top3=False)
        subgraphs = list(sorted(nx.connected_components(graph)))
        return CageGraph.from_atoms(ase.Atoms(picked_atoms[list(subgraphs[0])]))

    def cage(self) -> "FullereneCage":
        return self

    def cages(self) -> List["FullereneCage"]:
        return [self]

    def _check_atoms(self, atoms):
        # the cage should be one connected atoms
        graph = get_graph_from_atoms(atoms, only_top3=False)
        subgraphs = list(sorted(nx.connected_components(graph)))
        assert (
            len(subgraphs) == 1
        ), "The fullerene cage should be one connected atoms, got {}".format(subgraphs)
        return atoms

    def __str__(self):
        return "FullereneCage({})".format(
            self.atoms.get_chemical_formula(),
        )

    @cached_property
    def circle_finder(self):
        from fullerenetool.algorithm import dual_graph

        return dual_graph.py_graph_circle_finder(
            len(self.graph.graph.edges),
            np.array(self.graph.graph.edges).astype(np.int64).data,
        )

    @cached_property
    def circle_vertex_list(self) -> Iterable[np.ndarray]:
        return self.graph.circles()
