from functools import cached_property
from typing import Iterable

import ase
import networkx as nx

from fullerenetool.fullerene import BaseFullerene, FullereneCage
from fullerenetool.operator.graph import get_graph_from_atoms


class FullereneDerivative(BaseFullerene):
    """The fullerene cage with addons exogenous functionalization to
    the fullerene cage."""

    def __init__(self, atoms: ase.Atoms):
        """Basic class for fullerene objects.

        The BaseFullerene object is an abstract class that is used to define
        the basic structure of fullerene family.

        Args:
            atoms (ase.Atoms): The atoms that make up the fullerene.
        """
        self._atoms = atoms
        self._non_carbon_atoms = ase.Atoms(
            self.atoms[self.atoms.get_atomic_numbers() != 6]
        )
        self._init_check()

    def _init_check(self):
        # Check there is a cage
        self.cage

    @cached_property
    def cage(self) -> FullereneCage:
        # we assume the largest connected carbon graph is the cage
        carbon_atoms = ase.Atoms(self.atoms[self.atoms.get_atomic_numbers() == 6])
        graph = get_graph_from_atoms(carbon_atoms, only_top3=False)
        subgraphs = list(sorted(nx.connected_components(graph)))
        return FullereneCage.from_atoms(ase.Atoms(carbon_atoms[list(subgraphs[0])]))

    @cached_property
    def addons(self) -> Iterable[ase.Atoms]:
        graph = get_graph_from_atoms(self._non_carbon_atoms, only_top3=False)
        subgraphs = list(sorted(nx.connected_components(graph)))
        for subgraph in subgraphs:
            yield ase.Atoms(self.atoms[list(subgraph)])

    @property
    def atoms(self):
        return self._atoms
