from collections import defaultdict
from scipy.sparse import coo_matrix
import numpy as np

def linear_expr(variables, coefficients, const_coeff=0.):
    var2coeff = {v:c for v, c in zip(variables, coefficients)}
    const_coeff = const_coeff
    return LinearExpr(var2coeff, const_coeff)


class LinearExpr(object):
    def __init__(self, var2coeff, const_coeff):
        self.var2coeff = var2coeff
        self.const_coeff = const_coeff
    
    def variables_set(self):
        return set(self.var2coeff)
    
    def __add__(self, other):
        res = defaultdict(float)
        res.update(self.var2coeff)
        for v, c in other.var2coeff.items():
            res[v] += c
        return LinearExpr(var2coeff=dict(res), const_coeff=self.const_coeff + other.const_coeff)
        
    def __sub__(self, other):
        return self + (-1 * other)
    
    def __mul__(self, scalar):
        return LinearExpr(var2coeff={v:c*scalar for v, c in self.var2coeff.items()} ,const_coeff=self.const_coeff*scalar)
    
    def __rmul__(self, scalar):
        return self.__mul__(scalar)
    
    def __repr__(self):
        if self.const_coeff:
            l = ["{}".format(self.const_coeff)]
        else:
            l = []
        l += ["{:+} * {}".format(c, v) for v, c in self.var2coeff.items()]
        return " ".join(l)

def multidim_lin_expr(variables, A, b):
    n, m = A.shape
    assert b.shape == (n,)
    assert len(variables) == n
    lin_expr_list = []
    for i in range(n):
        lin_expr = linear_expr(variables, coefficients=A[i,:], const_coeff=b[i])
        lin_expr_list.append(lin_expr)
    return MultiDimLinearExpr(lin_expr_list)


class MultiDimLinearExpr(object):
    def __init__(self, lin_expr_list):
        self.lin_expr_list = lin_expr_list
        
    def variables_set(self):
        res = set()
        for lin_expr in self.lin_expr_list:
            res = res.union(lin_expr.variables_set())
        return res
        
    def __iter__(self):
        yield from iter(self.lin_expr_list)
    
    def __len__(self):
        return len(self.lin_expr_list)
    
    def extend(self, other):
        for lin_expr in other:
            self.lin_expr_list.append(lin_expr)
    
    def add(self, other):
        if self.lin_expr_list:
            self_len = len(self.lin_expr_list)
            other_len = len(other.lin_expr_list)

            if self_len != other_len:
                raise ValueError('programming error dimensions do not match. {} != {}'.format(self_len, other_len))
            
            lin_expr_list = [expr1 + expr2
                for expr1, expr2 in zip(self.lin_expr_list, other.lin_expr_list)]
            return MultiDimLinearExpr(lin_expr_list)
        else:
            return other # XXX potentially unsafe
        
    def append(self, lin_expr):
        self.lin_expr_list.append(lin_expr)
    
    def get_vars_and_matrices(self):
        n_eq = len(self.lin_expr_list)
        variables = sorted(self.variables_set())
        n_vars = len(variables)
        variable2index = {v:idx for idx, v in enumerate(variables)}
        A_i = []
        A_j = []
        A_data = []
        b = np.zeros((n_eq,))
        for i, lin_expr in enumerate(self.lin_expr_list):
            b[i] += lin_expr.const_coeff
            for v, coeff in lin_expr.var2coeff.items():
                j = variable2index[v]
                A_i.append(i)
                A_j.append(j)
                A_data.append(coeff)
        A = coo_matrix((A_data, (A_i, A_j)), shape=(n_eq, n_vars))
        A.eliminate_zeros()
        return variables, A, b

def main():
    expr1 = linear_expr(variables=['a', 'b'], coefficients=[0.1, -0.4])
    expr2 = linear_expr(variables=['b', 'c'], coefficients=[0.1, -4])
    print(expr1)
    print(expr2)
    print(expr1 + expr2)
    print(expr1 - expr2)

if __name__ == '__main__':
    main()

