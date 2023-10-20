import torch
import numpy as np

from expm32 import expm32
from expm64 import expm64

def differential(A, E, f):
    n = A.size(0)
    M = torch.zeros(2*n, 2*n, dtype=A.dtype, device=A.device)
    M[:n, :n] = A
    M[n:, n:] = A
    M[:n, n:] = E
    return f(M)[:n, n:]


class expm_pade_class(torch.autograd.Function):
    @staticmethod
    def _expm_func(A):
        if A.element_size() > 4:
            return expm64
        else:
            return expm32

    @staticmethod
    def forward(ctx, A):
        ctx.save_for_backward(A)
        expm = expm_pade_class._expm_func(A)
        return expm(A)

    @staticmethod
    def backward(ctx, G):
        (A,) = ctx.saved_tensors
        expm = expm_pade_class._expm_func(A)
        return differential(A.t(), G, expm)


expm = expm_pade_class.apply
