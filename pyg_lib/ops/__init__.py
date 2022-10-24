from typing import List

import torch
from torch import Tensor


class GroupedMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: List[Tensor], others: List[Tensor]):
        print("inside forward")
        ctx.save_for_backward(inputs, others)
        outs = torch.ops.pyg.grouped_matmul(inputs, others)

        return outs

    @staticmethod
    def backward(ctx, outs_grad: List[Tensor]):
        print('inside backward')
        inputs, others = ctx.saved_tensors

        inputs_grad = None
        if all([x.requires_grad for x in inputs]):
            for i in range(len(others)):
                others[i] = others[i].t()
            inputs_grad = torch.ops.pyg.grouped_matmul(outs_grad, others)
            for i in range(len(inputs)):
                inputs[i].grad = inputs_grad[i]

        others_grad = None
        if all([other.requires_grad for other in others]):
            print("Calculating other")
            quit()
            for i in range(len(inputs)):
                inputs[i] = inputs[i].t()
            others_grad = []
            # Considering GPU utilization, for-loops are actually preferred
            # here over the designated grouped matmul implementation:
            for i in range(len(inputs_t)):
                others[i].grad = inputs_t[i] @ outs_grad[i]


def grouped_matmul(inputs: List[Tensor], others: List[Tensor]) -> List[Tensor]:
    r"""Performs dense-dense matrix multiplication according to groups,
    utilizing dedicated kernels that effectively parallelize over groups.

    .. code-block:: python

        inputs = [torch.randn(5, 16), torch.randn(3, 32)]
        others = [torch.randn(16, 32), torch.randn(32, 64)]

        outs = pyg_lib.ops.grouped_matmul(inputs, others)
        assert len(outs) == 2
        assert outs[0].size() == (5, 32)
        assert outs[0] == inputs[0] @ others[0]
        assert outs[1].size() == (3, 64)
        assert outs[1] == inputs[1] @ others[1]

    Args:
        inputs (List[torch.Tensor]): List of left operand 2D matrices of shapes
            :obj:`[N_i, K_i]`.
        others (List[torch.Tensor]): List of right operand 2D matrices of
            shapes :obj:`[K_i, M_i]`.

    Returns:
        List[torch.Tensor]: List of 2D output matrices of shapes
        :obj:`[N_i, M_i]`.
    """

    outs = GroupedMatmul.apply(inputs, others)
    # NOTE Autograd doesnt set out[i].requires_grad = True automatically
    for src, other, out in zip(inputs, others, outs):
        out.requires_grad = src.requires_grad or other.requires_grad
    return out


def segment_matmul(inputs: Tensor, ptr: Tensor, other: Tensor) -> Tensor:
    r"""Performs dense-dense matrix multiplication according to segments along
    the first dimension of :obj:`inputs` as given by :obj:`ptr`, utilizing
    dedicated kernels that effectively parallelize over groups.

    .. code-block:: python

        inputs = torch.randn(8, 16)
        ptr = torch.tensor([0, 5, 8])
        other = torch.randn(2, 16, 32)

        out = pyg_lib.ops.segment_matmul(inputs, ptr, other)
        assert out.size() == (8, 32)
        assert out[0:5] == inputs[0:5] @ other[0]
        assert out[5:8] == inputs[5:8] @ other[1]

    Args:
        input (torch.Tensor): The left operand 2D matrix of shape
            :obj:`[N, K]`.
        ptr (torch.Tensor): Compressed vector of shape :obj:`[B + 1]`, holding
            the boundaries of segments.
            For best performance, given as a CPU tensor.
        other (torch.Tensor): The right operand 3D tensor of shape
            :obj:`[B, K, M]`.

    Returns:
        torch.Tensor: The 2D output matrix of shape :obj:`[N, M]`.
    """
    return torch.ops.pyg.segment_matmul(inputs, ptr, other)


__all__ = [
    'grouped_matmul',
    'segment_matmul',
]
