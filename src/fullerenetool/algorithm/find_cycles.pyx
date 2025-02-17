# distutils: language = c++
from cython.operator cimport dereference as deref
from libc.stdlib cimport free, malloc
from libcpp.vector cimport vector

import numpy as np

cimport numpy as np


cdef extern from "find_cycles.hpp" namespace "cycle_finder":
    cdef cppclass graph_cycle_finder:
        graph_cycle_finder(int edge_num, const long* edge_origin, int cores, bint verbose)
        vector[vector[unsigned long]] find_and_print_cycles()

cdef class py_graph_cycle_finder:
    cdef graph_cycle_finder* c_finder

    def __cinit__(self, int edge_num, np.ndarray[long, ndim=2] edge_origin, int cores=0, bint verbose=False):
        """
        Initialize the graph_cycle_finder with edge data.
        :param edge_num: Number of edges.
        :param edge_origin: A 2D numpy array where each row is [source, target].
        :param cores: Number of cores to use for parallel processing.
        :param verbose: Whether to print verbose log messages.
        """
        if edge_origin.shape[0] != edge_num or edge_origin.shape[1] != 2:
            raise ValueError("edge_origin must be a 2D array with shape (edge_num, 2)")

        # Ensure the array is contiguous and in C order
        edge_origin = np.ascontiguousarray(edge_origin, dtype=np.int64)

        # Create the C++ object
        self.c_finder = new graph_cycle_finder(edge_num, <long*> edge_origin.data, cores, verbose)

    def __dealloc__(self):
        if self.c_finder != NULL:
            del self.c_finder
            self.c_finder = NULL

    def find_and_print_cycles(self):
        """
        Find and print all cycles in the graph and return them.
        :return: List of cycles, each cycle is a list of node indices.
        """
        if self.c_finder is not NULL:
            cycles_cpp = self.c_finder.find_and_print_cycles()
            cycles_py = []
            for cycle in cycles_cpp:
                cycles_py.append([node for node in cycle])
            return cycles_py
        else:
            raise RuntimeError("Graph cycle finder has been deallocated")
