import numpy as np
from collections import namedtuple, defaultdict, Counter
import numbers
SolverMatrices = namedtuple('SolverMatrices', 'P, q, G, h, A, b')


def add_sv(d1, d2):
    d = defaultdict(float)
    for k, v in d1.items(): d[k] += v
    for k, v in d2.items(): d[k] += v
    return dict(d)

class Variable(object):
    def __init__(self, factory, name):
        self.factory = factory
        self.name = name

    def __eq__(self, other):
        return self.factory is other.factory and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __lt__(self, other):
        return str(self.name).__lt__(str(other.name))

    def __repr__(self):
        return str(self.name)

    def __str__(self):
        try:
            name, coords = self.name
            return "{}_{}".format('.'.join(name), coords)
        except:
            return  str(self.name)


class Term(object):
    def __init__(self, variables):
        self.variables = tuple(sorted(variables))

    def order(self):
        return len(self.variables)

    def __hash__(self):
        return hash(self.variables)

    def __eq__(self, other):
        return self.variables == other.variables

    def __lt__(self, other):
        assert isinstance(other, Term)
        self_order, other_order = self.order(), other.order()
        return self_order.__le__(other_order) or (self_order == other_order and self.variables.__le__(other.variables))

    def __repr__(self):
        return " ".join(map(str, sorted(self.variables)))

    def __str__(self):
        counter = Counter(self.variables)
        l = []
        for k in sorted(counter):
            exponent = counter[k]
            if exponent > 1:
                l.append("{}^{}".format(str(k), exponent))
            else:
                assert exponent == 1
                l.append(str(k))
        return " ".join(map(str, l))


class Expression(object):
    def __init__(self, factory, term2coeff=None):
        self.factory = factory
        if term2coeff is None:
            self.term2coeff = {}
        else:
            self.term2coeff = dict(term2coeff)

    def zero(self):
        return Expression(self.factory, {})

    def one(self):
        return Expression(self.factory, {Term(()):1.})

    def __mul__(self, elem):
        if isinstance(elem, numbers.Number):
            term2coeff = {t:c*elem for t, c in self.term2coeff.items()}
        elif isinstance(elem, Variable):
            term2coeff = {Term(t.variables + (elem,)):c for t, c in self.term2coeff.items()}
        else:
            raise ValueError("elem has type {} ----> {}".format(
                str(type(elem)),
                str(elem.shape)
            ))
        return Expression(self.factory, term2coeff)

    def __imul__(self, other):
        self.term2coeff = self.__mul__(other).term2coeff
        return  self

    def __add__(self, other):
        if self.factory is not other.factory: raise ValueError
        if isinstance(other, Expression):
            return Expression(self.factory, term2coeff=add_sv(self.term2coeff, other.term2coeff))
        elif isinstance(other, Variable):
            return Expression(self.factory, term2coeff=add_sv(self.term2coeff, {Term((other,)):1.}))
        elif isinstance(other, Term):
            return Expression(self.factory, term2coeff=add_sv(self.term2coeff, {other:1.}))

    def __iadd__(self, other):
        self.term2coeff = self.__add__(other).term2coeff
        return self

    def __repr__(self):
        return ' '.join("{:+} {}".format(coeff, term) for term, coeff in sorted(self.term2coeff.items(), key=lambda x:x[0]) if coeff != 0)

    def __str__(self):
        l = []
        for term, coeff in sorted(self.term2coeff.items(), key=lambda x:x[0]):
            if coeff != 0:
                if coeff == 1. and term.order() > 0:
                    l.append("+ {}".format(str(term)))
                else:
                    l.append("{:+} {}".format(coeff, str(term)))
        return ' '.join(l)

    def order(self):
        if self.term2coeff:
            return max([term.order() for term in self.term2coeff])
        else:
            return 0

    def add_term(self, term, coeff):
        assert isinstance(term, Term)
        for v in term.variables:
            self.factory.add_variable(v)

        if term not in self.term2coeff:
            self.term2coeff[term] = coeff
        else:
            self.term2coeff[term] += coeff

    def add_dotprod(self, x, z):
        for xi, zi in zip(x, z):
            elem_expr = self.one()
            elem_expr *= xi
            elem_expr *= zi
            self += elem_expr

    def add_quad(self, vars_v, coeff_mat, vars_w):
        n, m = len(vars_v), len(vars_w)
        assert coeff_mat.shape == (n, m)
        for i, v_i in enumerate(vars_v):
            for j, w_j in enumerate(vars_w):
                elem_expr = self.one()
                elem_expr *= coeff_mat[i, j]
                elem_expr *= v_i
                elem_expr *= w_j
                self += elem_expr

class QPBuilder(object):
    def __init__(self, dtype):
        self.dtype = dtype
        self.objective = None
        self.inequality_list = []
        self.equality_list = []
        self.var_set = set()

    def print_(self):
        return
        print('objective')
        print(self.objective)
        print('s.t.')
        for eq in self.equality_list:
            print(eq, '== 0')
        for ineq in self.inequality_list:
            print(ineq, '<= 0')

    def new_objective(self):
        if self.objective is not None:
            raise ValueError
        self.objective = self._create_expression()
        return self.objective

    def new_variable(self, name):
        variable = Variable(self, name)
        self.var_set.add(variable)
        return variable

    def var2idx(self):
        return {v:idx for idx, v in enumerate(self.variables())}

    def variables(self):
        return sorted(self.var_set)

    def add_variable(self, v):
        self.var_set.add(v)

    def _create_expression(self):
        return Expression(factory=self)

    def new_equality(self):
        expr = self._create_expression()
        self.equality_list.append(expr)
        return expr

    def new_inequality(self):
        # print('DEBUG INEQUALITIES')
        expr = self._create_expression()
        self.inequality_list.append(expr)
        return expr

    def new_vector(self, name, ndims):
        return [self.new_variable((name, i)) for i in range(ndims)]

    def Qp_matrices(self, variables, var2idx):
        n = len(variables)
        Q = np.zeros((n, n), dtype=self.dtype)
        p = np.zeros((n,), dtype=self.dtype)
        for term, coeff in self.objective.term2coeff.items():
            order = term.order()
            if order == 0:
                pass # ignore constants in the objective
            elif order == 1:
                x, = term.variables
                i_x = var2idx[x]
                p[i_x] += coeff
            elif order == 2:
                x, y = term.variables
                i_x = var2idx[x]
                i_y = var2idx[y]
                Q[i_x, i_y] += 2*coeff # XXX x2 because it will be divided by 2 in the objective
            else:
                raise ValueError('Term order higher than 2. TERM {}'.format(term))
        return Q, p

    def Ab_matrices(self, variables, var2idx):
        # print('Ab_matrices\n', '\n\t'.join(map(str, self.equality_list)))
        A, b = self.lin_constr2arrays(self.equality_list, variables, var2idx)
        return A, b

    def Gh_matrices(self, variables, var2idx):
        # print('Gh_matrices\n', '\n\t'.join(map(str, self.inequality_list)))
        G, h = self.lin_constr2arrays(self.inequality_list, variables, var2idx)
        return G, h

    def build(self):
        variables = self.variables()
        var2idx = self.var2idx()
        self.print_()
        # print('DEBUG variables', variables)
        # print('DEBUG var2idx', var2idx)
        P, q = self.Qp_matrices(variables, var2idx)
        A, b = self.Ab_matrices(variables, var2idx)
        G, h = self.Gh_matrices(variables, var2idx)
        return SolverMatrices(P, q, G, h, A, b)


    def lin_constr2arrays(self, constr_list, variables, var2idx):
        A_list = []
        b_list = []
        nvars = len(variables)
        for expr in constr_list:
            if expr.order() != 1:
                raise ValueError('Order {} constraints are not allowed {}'.format(expr.order(), str(expr)))
            row = np.zeros((nvars,), dtype=self.dtype)
            b = 0.
            for term, coeff in expr.term2coeff.items():
                if term.order() == 0:
                    b -= coeff
                elif term.order() == 1:
                    variable, = term.variables
                    vidx = var2idx[variable]
                    row[vidx] += coeff
                else:
                    raise ValueError
            A_list.append(row.reshape((1, -1)))
            b_list.append(b)
        if A_list:
            return np.concatenate(A_list), np.array(b_list, dtype=self.dtype)
        else:
            return None, None
