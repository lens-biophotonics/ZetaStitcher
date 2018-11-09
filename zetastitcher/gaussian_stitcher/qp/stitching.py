from .mapper import QPBuilder
from qpsolvers import solve_qp

from collections import defaultdict
import networkx as nx
import numpy as np


def get_node2coordinates(x, nodes, n_dims):
    node2coordinates = {}
    for idx, v in enumerate(nodes):
        node2coordinates[v] = x[idx*n_dims: (idx+1)*n_dims]
    return node2coordinates


class GaussianQPBuilder(object):
    def __init__(self, n_dims, digraph, v_origin):
        self.n_dims = n_dims
        self.digraph = digraph
        self.v_origin = v_origin
        self.builder = QPBuilder(dtype=np.double)
        self.node2coords = {
            v: self.builder.new_vector((v,), self.n_dims)
                for v in digraph.nodes()
        }
        self.edge2coords = {
            (v, w): self.builder.new_vector((v, w), self.n_dims)
                for v in digraph.nodes()
                    for w in digraph.successors(v)
        }

    def set_objective(self):
        objective = self.builder.new_objective()
        for v in self.digraph.nodes():
            for w in self.digraph.successors(v):
                data = self.digraph[v][w]
                Lambda_vw = data['Lambda']
                s_vw = self.edge2coords[(v, w)]
                objective.add_quad(s_vw, Lambda_vw, s_vw)

    def add_diff_equalities(self):
        for v in self.digraph.nodes():
            t_v = self.node2coords[v]
            for w in self.digraph.successors(v):
                t_w = self.node2coords[w]

                p_vw = self.digraph[v][w]['p']
                s_vw = self.edge2coords[(v, w)]
                # DIFF EQUALITIES
                for dim in range(self.n_dims):
                    diff_eq = self.builder.new_equality()
                    diff_eq.add_dotprod([t_w[dim], t_v[dim], p_vw[dim], s_vw[dim]], [1, -1, -1, -1])

    def add_orig_equalities(self):
        t_orig = self.node2coords[self.v_origin]
        for dim in range(self.n_dims):
            origin_eq = self.builder.new_equality()
            origin_eq.add_dotprod([t_orig[dim]], [1])

    def add_ub_inequalities(self):
        for v in self.digraph.nodes():
            t_v = self.node2coords[v]
            for w in self.digraph.successors(v):
                t_w = self.node2coords[w]
                ub = self.digraph[v][w].get('ub', None)
                if ub is not None:
                    for dim in range(self.n_dims):
                        ub_ineq = self.builder.new_inequality()
                        ub_ineq.add_dotprod([t_w[dim], t_v[dim], ub[dim]], [1, -1, -1])

    def add_lb_inequalities(self):
        for v in self.digraph.nodes():
            t_v = self.node2coords[v]
            for w in self.digraph.successors(v):
                t_w = self.node2coords[w]
                lb = self.digraph[v][w].get('lb', None)
                if lb is not None:
                    for dim in range(self.n_dims):
                        # lb <= t_w-t_v
                        # t_v-t_w + lb <= 0
                        lb_ineq = self.builder.new_inequality()
                        lb_ineq.add_dotprod([t_v[dim], t_w[dim], lb[dim]], [1, -1, 1])


    def build(self):
        return self.builder.build()

    def variables(self):
        return self.builder.variables()



class GaussianStitcherQP(object):
    def __init__(self, n_dims, solver):
        self.n_dims = n_dims
        self.solver = solver

    def stitch(self, data_in, v_origin):
        digraph = self._make_digraph(data_in)
        x, variables = self._optimize(digraph, v_origin)
        if x is None:
            raise Exception("solver error")
        x_sol = np.array(x).reshape((-1,))
        # print('VARS', variables)
        # print('VALS', x_sol)
        node2coords, _edge2coords = self._sol2coords(x_sol, variables)
        return node2coords, digraph

    def _sol2coords(self, x_sol, variables):
        assert len(variables) == len(x_sol)
        node2coords = defaultdict(lambda:np.zeros((self.n_dims,)))
        edge2coords = defaultdict(lambda:np.zeros((self.n_dims,)))
        for variable, value in zip(variables, x_sol):
            elem, coord_idx = variable.name
            if len(elem) == 1:
                node2coords[elem[0]][coord_idx] = value
            elif len(elem) == 2:
                edge2coords[elem][coord_idx] = value
            else:
                raise ValueError
        return dict(node2coords), dict(edge2coords)
#
    def _make_digraph(self, data_in):
        digraph = nx.DiGraph()
        for constraint_tuple in data_in:
            digraph.add_edge(
                constraint_tuple.v,
                constraint_tuple.w,
                Lambda=constraint_tuple.Lambda,
                p=constraint_tuple.p,
                lb=constraint_tuple.lb,
                ub=constraint_tuple.ub
            )
        return digraph
#
    def get_matrices(self, digraph, v_origin):
        gbuilder = GaussianQPBuilder(self.n_dims, digraph, v_origin)
        gbuilder.set_objective()
        gbuilder.add_diff_equalities()
        gbuilder.add_orig_equalities()
        gbuilder.add_lb_inequalities()
        gbuilder.add_ub_inequalities()
        solver_matrices = gbuilder.build()
        return solver_matrices, gbuilder.variables()

#
    def _optimize(self, digraph, v_origin):
        solver_matrices, variables = self.get_matrices(digraph, v_origin)
        P = solver_matrices.P
        q = solver_matrices.q
        G = solver_matrices.G
        h = solver_matrices.h
        A = solver_matrices.A
        b = solver_matrices.b

        # PAG  = np.concatenate([P, A, G])
        # print('rank(A)', np.linalg.matrix_rank(A))
        # print('shape(A)', A.shape)
        # print('rank([P; A; G])', np.linalg.matrix_rank(PAG))
        # print('shape([P; A; G])', PAG.shape)

        x = solve_qp(P, q, G, h, A, b, solver=self.solver)
        return x, variables
#
