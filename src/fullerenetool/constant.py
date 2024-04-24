import numpy as np
from ase.data import covalent_radii

# covalent_bond[i,j] = covalent[i] + covalent[j]
covalent_bond = np.zeros((len(covalent_radii), len(covalent_radii)))
for i in range(len(covalent_bond)):
    for j in range(i, len(covalent_bond)):
        covalent_bond[i, j] = max(covalent_radii[i] + covalent_radii[j], 0.5)
        covalent_bond[j, i] = covalent_bond[i, j]
