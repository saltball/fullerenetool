from functools import cached_property
from typing import List

import ase
import networkx as nx
import numpy as np
from ase.neighborlist import NeighborList, natural_cutoffs

from fullerenetool.fullerene import BaseFullerene
from fullerenetool.logger import logger
from fullerenetool.operator.calculator.bond_d3_calculator import BondCalculator
from fullerenetool.operator.graph import get_graph_from_atoms


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


class CageGraph(object):
    def __init__(
        self,
        adjacency_matrix: np.ndarray,
    ):
        """Initializes a CageGraph object.

        The CageGraph object is a representation of a fullerene graph. It
        assumes that the fullerene is a carbon cage with no restrictions on the
        number of atoms or the shape of the cage.

        All carbon atoms are connected to each other by bonds. The graph is
        represented by an adjacency matrix.

        Note the following for inputs:
        - The adjacency matrix is a square matrix that represents the bonds
          between atoms. The matrix is symmetric and the diagonal is zero.

        Args:
            adjacency_matrix (np.ndarray): The adjacency matrix of the cage.
        """
        if adjacency_matrix is None:
            raise ValueError("The adjacency_matrix is required.")
        self.adjacency_matrix = adjacency_matrix
        self.circle_matrix = self._get_circle_matrix()
        self._init_check()

    @cached_property
    def graph(self) -> nx.Graph:
        """The networkx graph object of the fullerene graph."""
        return nx.from_numpy_array(self.adjacency_matrix, create_using=nx.Graph)

    def _get_circle_matrix(self):
        from fullerenetool.algorithm import dual_graph

        circle_finder = dual_graph.py_graph_circle_finder(
            int(len(self.graph.edges)), np.array(self.graph.edges).astype(int).data
        )
        edge_list = circle_finder.get_dual_edge_list()
        circleADJ = np.zeros([circle_finder.face_size, circle_finder.face_size])
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

    @classmethod
    def from_carbon_cage(cls, atoms: ase.Atoms):
        """Creates a CageGraph object from an Carbon only ase.Atoms object."""
        fullerene = check_carbon_cage(atoms)
        return cls.from_atoms(fullerene)

    @classmethod
    def from_atoms(cls, fullerene: ase.Atoms):
        """Creates a CageGraph object from a ase.Atoms object."""
        fullerene_atoms = fullerene
        cutoffs = natural_cutoffs(fullerene_atoms)
        neighborList = NeighborList(cutoffs, self_interaction=False, bothways=True)
        neighborList.update(fullerene_atoms)
        adjacency_matrix = neighborList.get_connectivity_matrix(sparse=False)
        if (adjacency_matrix.sum(-1) != 3).any():
            adjacency_matrix = np.zeros_like(adjacency_matrix)
            logger.warning(
                "More than 3 neighbors by ase found for input atoms"
                "Trying using first 3 shortest distance."
            )
            neighborDis: np.ndarray = fullerene_atoms.get_all_distances()
            neighborSorted = np.argsort(neighborDis)
            adjacency_matrix[
                np.arange(len(fullerene))[:, None], neighborSorted[:, 1:4]
            ] = 1
        adjacency_matrix = np.array(adjacency_matrix)
        return CageGraph(adjacency_matrix=adjacency_matrix)

    def __str__(self) -> str:
        return "CageGraph object with {} nodes and {} edges".format(
            self.graph.number_of_nodes(), self.graph.number_of_edges()
        )

    def generate_atoms(
        self, elements: List[str], algorithm="d3", traj=None, max_steps=None
    ) -> ase.Atoms:
        """Generate atoms from the CageGraph.

        Args:
            elements (List[str]): The list of elements to be assigned to the atoms.

        Returns:
            ase.Atoms: The atoms structure of the CageGraph by certaining the
                elements.
        """
        # TODO(algorithm): implement generate atoms from CageGraph
        from ase import Atoms
        from ase.optimize.lbfgs import LBFGS

        calc = BondCalculator(
            topo=nx.adjacency_matrix(self.graph).todense(), device="cpu"
        )
        atoms = Atoms(
            symbols=elements,
            positions=[
                np.random.uniform(low=-1, high=1, size=3) for _ in range(len(elements))
            ],
            calculator=calc,
        )
        opt = LBFGS(atoms, trajectory=traj)
        opt.run(fmax=0.001, steps=max_steps)
        atoms.calc = None
        return atoms

        # raise NotImplementedError


class FullereneCage(BaseFullerene):
    """`FullereneCage` is a class that represents a fullerene cage.

    The cage means a large molecule that is formed by connecting atoms.
    It may contain carbon atoms and other atoms.
    """

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

    def _check_atoms(self, atoms):
        # the cage should be one connected atoms
        graph = get_graph_from_atoms(atoms, only_top3=False)
        subgraphs = list(sorted(nx.connected_components(graph)))
        assert (
            len(subgraphs) == 1
        ), "The fullerene cage should be one connected atoms, got {}".format(subgraphs)

    def __str__(self):
        return "FullereneCage({})".format(
            self.atoms.get_chemical_formula(),
        )
