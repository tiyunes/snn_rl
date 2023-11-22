from typing import Optional, Union, Tuple, List, Sequence, Iterable, Dict

import numpy as np
import torch

from bindsnet.learning import PostPre
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes, DiehlAndCookNodes
from bindsnet.network.topology import Connection, Conv2dConnection

            
            
class DiehlAndCook(Network):
    # language=rst
    """
    Implements the spiking neural network architecture from `(Diehl & Cook 2015)
    <https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full>`_.
    """

    def __init__(
        self,
        n_inpt: int,
        LearningRule = PostPre,
        n_neurons: int = 100,
        inh: float = 17.5,
        dt: float = 1.0,
        nu: Optional[Union[float, Sequence[float]]] = (1e-4, 1e-2),
        reduction: Optional[callable] = None,
        wmin: float = 0.0,
        wmax: float = 1.0,
        norm: float = 78.4,
        theta_plus: float = 0.05,
        tc_theta_decay: float = 1e7,
        lrnArgs : dict = {},
        excArgs : dict = {'rest':-65.0, 'reset':-65.0, 'thresh':-52.0, 'refrac':5, 'tc_decay':100.0, 'tc_trace':20.0},
        
        inpt_shape: Optional[Iterable[int]] = None,
    ) -> None:
       
        super().__init__(dt=dt)
        
        self.LearningRule = LearningRule
        for key in lrnArgs:
            if hasattr(self.LearningRule, key):
                setattr(self.LearningRule, key, lrnArgs[key])
       
        self.n_inpt = n_inpt
        self.inpt_shape = inpt_shape
        self.n_neurons = n_neurons
        self.inh = inh
        self.dt = dt
        
        input_layer = Input(
            n=self.n_inpt, shape=self.inpt_shape, traces=True, tc_trace=excArgs['tc_trace'], trace_scale=1.0,
        )
        self.add_layer(input_layer, name="X")

        output_layer = DiehlAndCookNodes(
            n=self.n_neurons,
            traces=True,
            theta_plus=theta_plus,
            tc_theta_decay=tc_theta_decay,
            **excArgs,
        )
        self.add_layer(output_layer, name="Y")

        w = 0.3 * torch.rand(self.n_inpt, self.n_neurons)
        input_connection = Connection(
            source=self.layers["X"],
            target=self.layers["Y"],
            w=w,
            update_rule=self.LearningRule,
            nu=nu,
            reduction=reduction,
            wmin=wmin,
            wmax=wmax,
            norm=norm,
        )
        self.add_connection(input_connection, source="X", target="Y")

        w = -self.inh * (
            torch.ones(self.n_neurons, self.n_neurons)
            - torch.diag(torch.ones(self.n_neurons))
        )
        recurrent_connection = Connection(
            source=self.layers["Y"],
            target=self.layers["Y"],
            w=w,
            wmin=-self.inh,
            wmax=0,
        )
        self.add_connection(recurrent_connection, source="Y", target="Y")
        
class convDiehlAndCook(Network):
    
    def __init__(self,
        dims: Sequence[float],
        LearningRule = PostPre,
        n_filters: int = 25,
        kernel_size: int = 28,
        padding: int = 0,
        stride: int = 4,
        inh: float = 100,
        dt: float = 1.0,
        nu: Optional[Union[float, Sequence[float]]] = (1e-4, 1e-2),
        reduction: Optional[callable] = None,
        wmin: float = 0.0,
        wmax: float = 1.0,
        norm: float = 78.4,
        theta_plus: float = 0.05,
        tc_theta_decay: float = 1e7,
        lrnArgs : dict = {},
        excArgs : dict = {'rest':-65.0, 'reset':-65.0, 'thresh':-52.0, 'refrac':5, 'tc_decay':100.0, 'tc_trace':20.0},
        )->None:
        
        super().__init__(dt=dt)
        
        self.LearningRule = LearningRule
        for key in lrnArgs:
            if hasattr(self.LearningRule, key):
                setattr(self.LearningRule, key, lrnArgs[key])
                
        self.dims = dims
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        
        conv_size_w = int((dims[1] - kernel_size + 2 * padding) / stride) + 1
        conv_size_h = int((dims[0] - kernel_size + 2 * padding) / stride) + 1
        
        self.conv_size_w = conv_size_w
        self.conv_size_h = conv_size_h
        

        # Build network.
        input_layer = Input(n=dims[0]*dims[1], shape=(1, dims[0], dims[1]), traces=True)

        conv_layer = DiehlAndCookNodes(
            n=n_filters * conv_size_h * conv_size_w,
            shape=(n_filters, conv_size_h, conv_size_w),
            traces=True,
            theta_plus=theta_plus,
            tc_theta_decay=tc_theta_decay,
            **excArgs,
        )

        conv_conn = Conv2dConnection(
            input_layer,
            conv_layer,
            kernel_size=kernel_size,
            stride=stride,
            update_rule=self.LearningRule,
            norm=norm * kernel_size ** 2,
            nu=nu,
            wmin = wmin,
            wmax=wmax,
        )

        w = torch.zeros(n_filters, conv_size_h, conv_size_w, n_filters, conv_size_h, conv_size_w)
        for fltr1 in range(n_filters):
            for fltr2 in range(n_filters):
                if fltr1 != fltr2:
                    for i in range(conv_size_h):
                        for j in range(conv_size_w):
                            w[fltr1, i, j, fltr2, i, j] = -inh

        w = w.view(n_filters * conv_size_h * conv_size_w, n_filters * conv_size_h * conv_size_w)
        recurrent_conn = Connection(conv_layer, conv_layer, w=w)

        self.add_layer(input_layer, name="X")
        self.add_layer(conv_layer, name="Y")
        self.add_connection(conv_conn, source="X", target="Y")
        self.add_connection(recurrent_conn, source="Y", target="Y")
        