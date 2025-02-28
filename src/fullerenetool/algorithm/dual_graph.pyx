# -*- coding: utf-8 -*-
# ====================================== #
# @Author  : Yanbo Han
# @Email   : yanbohan98@gmail.com
# @File    : dual.pyx
# ALL RIGHTS ARE RESERVED UNLESS STATED.
# ====================================== #

import numpy as np

cimport numpy as np
from libcpp.vector cimport vector

import numpy as np

from libcpp.vector cimport vector


cdef extern from "dual_graph.hpp" namespace "dual_graph":
    cdef cppclass graph_circle_finder:
        int edge_num # Size of edges, not vertex!
        long *edge_origin # Array of edges.
        int planar_flag # If the graph is planarity or not.
        graph_circle_finder(int edge_num, long* edge_origin) except +
        int privacy_get_graph_face_num() # Number of faces.
        int privacy_get_face_vertex_num_sum() # Sum of vertex number in all faces. (Length of vertex_list and edge_list.)
        int privacy_get_dual_edge_num() # Number of edges in dual graph.
        void get_circle_vertex_num_list(vector[int]* vertex_num_list)
        void get_circle_edge_num_list(vector[int]* edge_num_list)
        void get_circle_vertex_list(vector[int]* vertex_num_list,vector[int]* vertex_list)
        void get_circle_edge_list(vector[int]* edge_num_list,vector[vector[int]]* edge_list)
        void get_dual_edge_list(int dual_edge_num, vector[vector[int]]* edge_list)

cdef class py_graph_circle_finder:
    cdef graph_circle_finder *c_finder
    # cdef void * _data_vector_e
    # cdef void * _data_vector_v
    # cdef void * _data_vector_dual_edge

    def __cinit__(self, int edge_num, long[:,:] edge_origin):
        try:
            self.c_finder = new graph_circle_finder(edge_num, &edge_origin[0,0])
        except Exception:
            print("Error: graph_circle_finder __cinit__")
            raise

    def __dealloc__(self):
        del self.c_finder

    def get_face_vertex_list(self):
        cdef vector[int]* vertex_num_list=new vector[int](self.c_finder.privacy_get_graph_face_num())
        self.c_finder.get_circle_vertex_num_list(&vertex_num_list[0])
        cdef vector[int]* vertex_list=new vector[int](self.c_finder.privacy_get_face_vertex_num_sum())
        self.c_finder.get_circle_vertex_list(&vertex_num_list[0], &vertex_list[0])
        face_v_slice,face_v= (np.asarray(<vector[int]&> vertex_num_list[0]), np.asarray(<vector[int]&> vertex_list[0]))
        del vertex_num_list
        del vertex_list
        return list(face_v[face_v_slice[:idx].sum():face_v_slice[:idx+1].sum()] for idx in range(self.face_size))

    def get_dual_edge_list(self):
        cdef vector[vector[int]]* dual_edge_list=new vector[vector[int]](self.c_finder.privacy_get_dual_edge_num(),vector[int](2,0))
        self.c_finder.get_dual_edge_list(self.dual_size, &dual_edge_list[0])
        dual_edge= np.asarray(<vector[vector[int]]&> dual_edge_list[0])
        del dual_edge_list
        return list(dual_edge)

    @property
    def face_size(self):
        return self.c_finder.privacy_get_graph_face_num()
    @property
    def dual_size(self):
        return self.c_finder.privacy_get_dual_edge_num()
