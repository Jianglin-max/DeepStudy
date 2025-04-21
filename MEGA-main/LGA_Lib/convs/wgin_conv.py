from typing import Callable, Union
import torch
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import OptPairTensor, Adj, Size
from LGA_Lib.convs.inits import reset

class WGINConv(MessagePassing):
    def __init__(
        self,
        nn: Callable,
        eps: float = 0.0,
        train_eps: bool = False,
        **kwargs
    ):
        kwargs.setdefault("aggr", "add")
        super(WGINConv, self).__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer("eps", torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_weight = None,  # 外部调用时仍使用 `edge_weight` 参数名
        size: Size = None
    ) -> Tensor:
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # 关键修改：将 `edge_weight` 映射为 PyG 预期的 `edge_attr`
        out = self.propagate(
            edge_index, x=x, edge_attr=edge_weight, size=size  # 参数名统一为 edge_attr
        )

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor, edge_attr) -> Tensor:  # 参数名统一为 edge_attr
        # edge_attr 可能为 None（未提供边权重）
        return x_j if edge_attr is None else x_j * edge_attr.view(-1, 1)

    def __repr__(self):
        return f"{self.__class__.__name__}(nn={self.nn})"

class WGINConv_sd(MessagePassing):
    def __init__(
        self,
        eps: float = 0.0,
        train_eps: bool = False,
        **kwargs
    ):
        kwargs.setdefault("aggr", "add")
        super(WGINConv_sd, self).__init__(**kwargs)
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer("eps", torch.Tensor([eps]))

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_weight = None,
        size: Size = None
    ) -> Tensor:
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # 关键修改：传递 edge_attr
        out = self.propagate(
            edge_index, x=x, edge_attr=edge_weight, size=size  # 参数名统一为 edge_attr
        )

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return out

    def message(self, x_j: Tensor, edge_attr) -> Tensor:  # 参数名统一为 edge_attr
        return x_j if edge_attr is None else x_j * edge_attr.view(-1, 1)

    def __repr__(self):
        return f"{self.__class__.__name__}()"  # 修复：移除对 self.nn 的引用
