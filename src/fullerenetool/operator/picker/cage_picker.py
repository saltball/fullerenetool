import warnings

import ase
import networkx as nx

from fullerenetool.operator.graph import get_graph_from_atoms


def cage_picker(atoms: ase.Atoms, atom_number=6) -> ase.Atoms:
    # we assume the largest connected carbon graph is the cage
    carbon_atoms = ase.Atoms(atoms[atoms.get_atomic_numbers() == atom_number])
    graph = get_graph_from_atoms(carbon_atoms, only_top3=False)
    subgraphs = list(sorted(nx.connected_components(graph)))
    if len(subgraphs[0]) - len(subgraphs[1]) < 10:
        warnings.warn(
            "The cage size is close to the second largest connected component,"
            "{} vs {}, be careful.".format(len(subgraphs[0]), len(subgraphs[1]))
        )
    return ase.Atoms(carbon_atoms[list(subgraphs[0])])
