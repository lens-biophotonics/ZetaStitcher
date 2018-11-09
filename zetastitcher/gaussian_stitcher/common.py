import numpy as np


def ConstraintTuple(ndims, eps_reg=None):
    class ConstraintTupleND(object):
        __slots__ = ['v', 'w', 'Lambda', 'p', 'ub', 'lb']
        def __init__(self, v, w, Lambda, p, ub=None, lb=None):
            self.v = v
            self.w = w
            self.Lambda = Lambda
            self.p = p
            self.ub = ub
            self.lb = lb
            self._check_dims(Lambda, p, ub, lb)
            if eps_reg is not None:
                if eps_reg < 0:
                    raise ValueError('eps_reg < 0')
                self.Lambda += eps_reg * np.eye(Lambda.shape[0])

        def _check_dims(self, Lambda, p, ub, lb):
            if Lambda.shape != (ndims, ndims):
                raise ValueError('Lambda.shape should be {} rather than {}.'.format((ndims, ndims), Lambda.shape))
            if p.shape != (ndims,):
                raise ValueError('p.shape should be {} rather than {}.'.format((ndims,), p.shape))

            if ub is not None and ub.shape != (ndims,):
                raise ValueError('ub.shape should be {} rather than {}.'.format((ndims,), ub.shape))

            if lb is not None and lb.shape != (ndims,):
                raise ValueError('lb.shape should be {} rather than {}.'.format((ndims,), lb.shape))
    return ConstraintTupleND
