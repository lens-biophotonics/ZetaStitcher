from .mapper import MultiDimLinearExpr, multidim_lin_expr, linear_expr

from collections import defaultdict
import networkx as nx
import numpy as np


class GaussianStitcher(object):
    def __init__(self, n_dims, solver_hook):
        self.n_dims = n_dims
        self.solver_hook = solver_hook
    
    def stitch(self, data_in, v_origin):
        digraph = self._make_digraph(data_in)
        # print('digraph.number_of_nodes()', digraph.number_of_nodes())
        # print('digraph.nodes()', digraph.nodes())
        # print('digraph.number_of_edges()', digraph.number_of_edges())
        lin_expressions = self._make_constraints(digraph, v_origin)
        node2coordinates = self._optimize(lin_expressions)
        return node2coordinates, digraph

    def _make_digraph(self, data_in):
        digraph = nx.DiGraph()
        
        
        for constraint_tuple in data_in:
            if constraint_tuple.lb is not None:
                raise ValueError('The linear solver cannot handle lower bound constraints (lb must be None)')
            if constraint_tuple.ub is not None:
                raise ValueError('The linear solver cannot handle upper bound constraints (ub must be None)')
            digraph.add_edge(
                constraint_tuple.v,
                constraint_tuple.w,
                Lambda=constraint_tuple.Lambda,
                p=constraint_tuple.p
            )
        return digraph
    
    def _make_constraints(self, digraph, v_origin):
        # ADD GAUSSIAN POTENTIAL
        lin_expressions = MultiDimLinearExpr([])
        for u in digraph.nodes():
            multidim_lin_expr_acc = MultiDimLinearExpr([])
            for v in digraph.nodes():
                for w in digraph.successors(v):
                    if v == w:
                        raise ValueError('Self loops are not allowed. Check vertex {}.'.format(v))
                    if u == v or u == w:
                        delta_vwu = 1. if u == v else -1.
                        p_vw = digraph[v][w]['p']
                        Lambda_vw = digraph[v][w]['Lambda']
                        current_lin_expr = multidim_lin_expr(
                            variables = [('s', v, w, dim_i) for dim_i in range(self.n_dims)],
                            A = delta_vwu * Lambda_vw,
                            b = - delta_vwu * np.dot(Lambda_vw, p_vw)
                        )
                        multidim_lin_expr_acc = multidim_lin_expr_acc.add(current_lin_expr)
                    else:
                        pass
            lin_expressions.extend(multidim_lin_expr_acc)
    
        # ADD CONSTRAINTS ON SLACKS
        for v, w in digraph.edges():
            for dim_i in range(self.n_dims):
                lin_expressions.append(
                    linear_expr(
                        variables = [
                            ('s', v, w, dim_i),
                            ('t', v, dim_i),
                            ('t', w, dim_i)
                        ],
                        coefficients = [
                            1.,
                            -1.,
                            1.
                        ]
                    )
                )
        # ADD CONSTRAINT FOR THE ORIGIN
        if not digraph.has_node(v_origin):
            raise ValueError('Origin node {} is missing!!'.format(v_origin))

        for dim_i in range(self.n_dims):
            lin_expressions.append(
                linear_expr(
                    variables = [('t', v_origin, dim_i)],
                    coefficients = [1.]
                )
            )
        return lin_expressions

    def _optimize(self, lin_expressions):
        variables, A, b = lin_expressions.get_vars_and_matrices()
        x = self.solver_hook(A, b)
        var2value = {vi:xi for vi, xi in zip(variables, x)}
        node2coordinates = get_node2coordinates(var2value, n_dims=self.n_dims)
        return node2coordinates


def get_node2coordinates(var2value, n_dims):
    node2coordinates = defaultdict(lambda:np.zeros((n_dims,)))
    for var, value in var2value.items():
        if var[0] == 't':
            _, v, dim_i = var
            node2coordinates[v][dim_i] = value
    return node2coordinates
