from typing import Iterable

import ase
import deprecation
import networkx as nx

from fullerenetool import __version__
from fullerenetool.operator.graph import get_graph_from_atoms


@deprecation.deprecated(
    deprecated_in="0.0.1",
    removed_in="0.0.2",
    current_version=__version__,
    details="Use the DerivativeFullerene class instead",
)
def get_cage_from_fullerene_derivative(atoms) -> ase.Atoms:
    # we assume the largest connected carbon graph is the cage
    carbon_atoms = ase.Atoms(atoms[atoms.get_atomic_numbers() == 6])
    graph = get_graph_from_atoms(carbon_atoms, only_top3=False)
    subgraphs = list(sorted(nx.connected_components(graph)))
    return ase.Atoms(carbon_atoms[list(subgraphs[0])])


@deprecation.deprecated(
    deprecated_in="0.0.1",
    removed_in="0.0.2",
    current_version=__version__,
    details="Use the DerivativeFullerene class instead",
)
def addons_of_fullerene_derivative(atoms) -> Iterable[ase.Atoms]:
    _non_carbon_atoms = ase.Atoms(atoms[atoms.get_atomic_numbers() != 6])
    graph = get_graph_from_atoms(_non_carbon_atoms, only_top3=False)
    subgraphs = list(sorted(nx.connected_components(graph)))
    for subgraph in subgraphs:
        yield ase.Atoms(atoms[list(subgraph)])


# class BaseDerivativeFullerene(BaseFullerene):
#     def __init__(self, atoms, cage_graph: CageGraph):
#         self._atoms = atoms
#         self.cage_graph = cage_graph
#         self.derivatives =
