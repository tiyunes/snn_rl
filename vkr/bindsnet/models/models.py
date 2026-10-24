from typing import Optional, Union, Tuple, List, Sequence, Iterable

import numpy as np
import os
import torch
from scipy.spatial.distance import euclidean
from torch.nn.modules.utils import _pair
import torch.nn as nn
from torchvision import models

from ..learning import PostPre, MSTDPET, NoOp, TDLTP, FBTDSTDP
from ..network import Network, load
from ..network.nodes import Input, LIFNodes, DiehlAndCookNodes, DiehlAndCookNodesModified
from ..network.topology import Connection, LocalConnection


class TwoLayerNetwork(Network):
    # language=rst
    """
    Implements an ``Input`` instance connected to a ``LIFNodes`` instance with a
    fully-connected ``Connection``.
    """

    def __init__(
        self,
        n_inpt: int,
        n_neurons: int = 100,
        dt: float = 1.0,
        wmin: float = 0.0,
        wmax: float = 1.0,
        nu: Optional[Union[float, Sequence[float]]] = (1e-4, 1e-2),
        reduction: Optional[callable] = None,
        norm: float = 78.4,
    ) -> None:
        # language=rst
        """
        Constructor for class ``TwoLayerNetwork``.

        :param n_inpt: Number of input neurons. Matches the 1D size of the input data.
        :param n_neurons: Number of neurons in the ``LIFNodes`` population.
        :param dt: Simulation time step.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events,
            respectively.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param wmin: Minimum allowed weight on ``Input`` to ``LIFNodes`` synapses.
        :param wmax: Maximum allowed weight on ``Input`` to ``LIFNodes`` synapses.
        :param norm: ``Input`` to ``LIFNodes`` layer connection weights normalization
            constant.
        """
        super().__init__(dt=dt)

        self.n_inpt = n_inpt
        self.n_neurons = n_neurons
        self.dt = dt

        self.add_layer(Input(n=self.n_inpt, traces=True, tc_trace=20.0), name="X")
        self.add_layer(
            LIFNodes(
                n=self.n_neurons,
                traces=True,
                rest=-65.0,
                reset=-65.0,
                thresh=-52.0,
                refrac=5,
                tc_decay=100.0,
                tc_trace=20.0,
            ),
            name="Y",
        )

        w = 0.3 * torch.rand(self.n_inpt, self.n_neurons)
        self.add_connection(
            Connection(
                source=self.layers["X"],
                target=self.layers["Y"],
                w=w,
                update_rule=PostPre,
                nu=nu,
                reduction=reduction,
                wmin=wmin,
                wmax=wmax,
                norm=norm,
            ),
            source="X",
            target="Y",
        )


class DiehlAndCook2015(Network):
    # language=rst
    """
    Implements the spiking neural network architecture from `(Diehl & Cook 2015)
    <https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full>`_.
    """

    def __init__(
        self,
        n_inpt: int,
        n_neurons: int = 100,
        exc: float = 22.5,
        inh: float = 17.5,
        dt: float = 1.0,
        nu: Optional[Union[float, Sequence[float]]] = (1e-4, 1e-2),
        reduction: Optional[callable] = None,
        wmin: float = 0.0,
        wmax: float = 1.0,
        norm: float = 78.4,
        theta_plus: float = 0.05,
        tc_theta_decay: float = 1e7,
        inpt_shape: Optional[Iterable[int]] = None,
    ) -> None:
        # language=rst
        """
        Constructor for class ``DiehlAndCook2015``.

        :param n_inpt: Number of input neurons. Matches the 1D size of the input data.
        :param n_neurons: Number of excitatory, inhibitory neurons.
        :param exc: Strength of synapse weights from excitatory to inhibitory layer.
        :param inh: Strength of synapse weights from inhibitory to excitatory layer.
        :param dt: Simulation time step.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events,
            respectively.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param wmin: Minimum allowed weight on input to excitatory synapses.
        :param wmax: Maximum allowed weight on input to excitatory synapses.
        :param norm: Input to excitatory layer connection weights normalization
            constant.
        :param theta_plus: On-spike increment of ``DiehlAndCookNodes`` membrane
            threshold potential.
        :param tc_theta_decay: Time constant of ``DiehlAndCookNodes`` threshold
            potential decay.
        :param inpt_shape: The dimensionality of the input layer.
        """
        super().__init__(dt=dt)

        self.n_inpt = n_inpt
        self.inpt_shape = inpt_shape
        self.n_neurons = n_neurons
        self.exc = exc
        self.inh = inh
        self.dt = dt

        # Layers
        input_layer = Input(
            n=self.n_inpt, shape=self.inpt_shape, traces=True, tc_trace=20.0
        )
        exc_layer = DiehlAndCookNodes(
            n=self.n_neurons,
            traces=True,
            rest=-65.0,
            reset=-60.0,
            thresh=-52.0,
            refrac=5,
            tc_decay=100.0,
            tc_trace=20.0,
            theta_plus=theta_plus,
            tc_theta_decay=tc_theta_decay,
        )

        inh_layer = LIFNodes(
            n=self.n_neurons,
            traces=False,
            rest=-60.0,
            reset=-45.0,
            thresh=-40.0,
            tc_decay=10.0,
            refrac=2,
            tc_trace=20.0,
        )

        # Connections
        w = 0.3 * torch.rand(self.n_inpt, self.n_neurons)
        input_exc_conn = Connection(
            source=input_layer,
            target=exc_layer,
            w=w,
            update_rule=PostPre,
            nu=nu,
            reduction=reduction,
            wmin=wmin,
            wmax=wmax,
            norm=norm,
        )
        w = self.exc * torch.diag(torch.ones(self.n_neurons))
        exc_inh_conn = Connection(
            source=exc_layer, target=inh_layer, w=w, wmin=0, wmax=self.exc
        )
        w = -self.inh * (
            torch.ones(self.n_neurons, self.n_neurons)
            - torch.diag(torch.ones(self.n_neurons))
        )
        inh_exc_conn = Connection(
            source=inh_layer, target=exc_layer, w=w, wmin=-self.inh, wmax=0
        )

        # Add to network
        self.add_layer(input_layer, name="X")
        self.add_layer(exc_layer, name="Ae")
        self.add_layer(inh_layer, name="Ai")
        self.add_connection(input_exc_conn, source="X", target="Ae")
        self.add_connection(exc_inh_conn, source="Ae", target="Ai")
        self.add_connection(inh_exc_conn, source="Ai", target="Ae")


class DiehlAndCook2015v2(Network):
    # language=rst
    """
    Slightly modifies the spiking neural network architecture from `(Diehl & Cook 2015)
    <https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full>`_ by removing
    the inhibitory layer and replacing it with a recurrent inhibitory connection in the
    output layer (what used to be the excitatory layer).
    """

    def __init__(
        self,
        n_inpt: int,
        n_neurons: int = 100,
        inh: float = 17.5,
        dt: float = 1.0,
        nu: Optional[Union[float, Sequence[float]]] = (1e-4, 1e-2),
        reduction: Optional[callable] = None,
        wmin: Optional[float] = 0.0,
        wmax: Optional[float] = 1.0,
        norm: float = None,
        theta_plus: float = 0.05,
        tc_theta_decay: float = 1e7,
        inpt_shape: Optional[Iterable[int]] = None,
        w_init_hid = 0.6,
        thresh=-52.,
        refrac=5,
        initial_model_path=None,
        one_spike_output=False,
    ) -> None:
        # language=rst
        """
        Constructor for class ``DiehlAndCook2015v2``.

        :param n_inpt: Number of input neurons. Matches the 1D size of the input data.
        :param n_neurons: Number of excitatory, inhibitory neurons.
        :param inh: Strength of synapse weights from inhibitory to excitatory layer.
        :param dt: Simulation time step.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events,
            respectively.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param wmin: Minimum allowed weight on input to excitatory synapses.
        :param wmax: Maximum allowed weight on input to excitatory synapses.
        :param norm: Input to excitatory layer connection weights normalization
            constant.
        :param theta_plus: On-spike increment of ``DiehlAndCookNodes`` membrane
            threshold potential.
        :param tc_theta_decay: Time constant of ``DiehlAndCookNodes`` threshold
            potential decay.
        :param inpt_shape: The dimensionality of the input layer.
        """
        super().__init__(dt=dt)

        self.n_inpt = n_inpt
        self.inpt_shape = inpt_shape
        self.n_neurons = n_neurons
        self.inh = inh
        self.dt = dt

        self.w_init_hid = w_init_hid
        self.thresh = thresh
        self.refrac = refrac

        self.one_spike_output = one_spike_output

        input_layer = Input(
            n=self.n_inpt, shape=self.inpt_shape, traces=True, tc_trace=20.0
        )
        self.add_layer(input_layer, name="X")

        output_layer = DiehlAndCookNodes(
            n=self.n_neurons,
            traces=True,
            rest=-65.0,
            reset=-65.0, # default: -60
            thresh=self.thresh, # default: -58
            refrac=self.refrac,
            tc_decay=15.0, # default: 100
            tc_trace=10.0, # default: 20
            theta_plus=0, # default: theta_plus
            tc_theta_decay=tc_theta_decay,
            one_spike=self.one_spike_output,
        )
        self.add_layer(output_layer, name="Y")

        if initial_model_path is None:
            w = self.w_init_hid * torch.rand(self.n_inpt, self.n_neurons)
        else:
            w = torch.load(initial_model_path).connections[('X', 'Y')].w

        input_connection = Connection(
            source=self.layers["X"],
            target=self.layers["Y"],
            w=w,
            update_rule=PostPre,
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

class DiehlOneLayerNonSpikingInput(Network):
    # language=rst
    """
    first layer of Diehl network with non-spiking input 
    """

    def __init__(
        self,
        n_inpt: int,
        n_neurons: int = 100,
        inh: float = 17.5,
        dt: float = 1.0,
        nu: Optional[Union[float, Sequence[float]]] = (1e-4, 1e-2),
        reduction: Optional[callable] = None,
        wmin: Optional[float] = 0.0,
        wmax: Optional[float] = 1.0,
        norm: float = None,
        theta_plus: float = 0.05,
        tc_theta_decay: float = 1e7,
        inpt_shape: Optional[Iterable[int]] = None,
        w_init_hid = 0.6,
        thresh=-52.,
        refrac=5,
    ) -> None:
        # language=rst
        super().__init__(dt=dt)

        self.n_inpt = n_inpt
        self.inpt_shape = inpt_shape
        self.n_neurons = n_neurons
        self.inh = inh
        self.dt = dt

        self.w_init_hid = w_init_hid
        self.thresh = thresh
        self.refrac = refrac

        input_layer = Input(
            n=self.n_inpt, shape=self.inpt_shape, traces=True, tc_trace=20.0
        )
        self.add_layer(input_layer, name="X")

        output_layer = DiehlAndCookNodes(
            n=self.n_neurons,
            traces=True,
            rest=-65.0,
            reset=-65.0, # default: -60
            thresh=self.thresh, # default: -58
            refrac=self.refrac,
            tc_decay=15.0, # default: 100
            tc_trace=10.0, # default: 20
            theta_plus=0, # default: theta_plus
            tc_theta_decay=tc_theta_decay,
            one_spike=False,
        )
        self.add_layer(output_layer, name="Y")

        w = self.w_init_hid * torch.rand(self.n_inpt, self.n_neurons)
        input_connection = Connection(
            source=self.layers["X"],
            target=self.layers["Y"],
            w=w,
            update_rule=PostPre,
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

class Diehl2Layers(Network):
    # language=rst
    """
    Slightly modifies the spiking neural network architecture from `(Diehl & Cook 2015)
    <https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full>`_ by removing
    the inhibitory layer and replacing it with a recurrent inhibitory connection in the
    output layer (what used to be the excitatory layer).
    """

    def __init__(
        self,
        n_inpt: int = 60,
        n_neurons_hid: int = 100,
        n_neurons_out: int = 2,
        inh_hid: float = 0.1,
        inh_out: float = 17.5,
        dt: float = 1.0,
        nu_hid: Optional[Union[float, Sequence[float]]] = (1e-4, 1e-2),
        nu_out: Optional[Union[float, Sequence[float]]] = (1e-4, 1e-2),
        reduction: Optional[callable] = None,
        wmin: Optional[float] = 0.0,
        wmax: Optional[float] = 1.0,
        norm: float = None,
        theta_plus: float = 0.05,
        tc_theta_decay: float = 1e7,
        inpt_shape: Optional[Iterable[int]] = None,
        hidden_layer_path: str = None,
        w_init_hid=0.5,
        w_init_out=0.5,
        thresh_hid=-58.0,
        thresh_out=-61.0,
        refrac_hid=5,
        refrac_out=1,
        one_spike_hid=False,
        one_spike_out=False,
        tc_trace_input=20.0,
        tc_decay_hid=15.0,
        tc_decay_out=15.0,
        tc_trace_hid=10.0,
        tc_trace_out=10.0,
        tc_plus_out=10.0,
        tc_minus_out=10.0,
        tc_e_trace_out=15.0,
    ) -> None:
        # language=rst
        """
        Constructor for class ``DiehlAndCook2015v2``.

        :param n_inpt: Number of input neurons. Matches the 1D size of the input data.
        :param n_neurons: Number of excitatory, inhibitory neurons.
        :param inh: Strength of synapse weights from inhibitory to excitatory layer.
        :param dt: Simulation time step.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events,
            respectively.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param wmin: Minimum allowed weight on input to excitatory synapses.
        :param wmax: Maximum allowed weight on input to excitatory synapses.
        :param norm: Input to excitatory layer connection weights normalization
            constant.
        :param theta_plus: On-spike increment of ``DiehlAndCookNodes`` membrane
            threshold potential.
        :param tc_theta_decay: Time constant of ``DiehlAndCookNodes`` threshold
            potential decay.
        :param inpt_shape: The dimensionality of the input layer.
        """
        super().__init__(dt=dt)

        self.n_inpt = n_inpt
        self.inpt_shape = inpt_shape
        self.n_neurons_hid = n_neurons_hid
        self.n_neurons_out = n_neurons_out
        self.inh_hid = inh_hid
        self.inh_out = inh_out
        self.dt = dt

        self.w_init_hid = w_init_hid
        self.w_init_out = w_init_out
        
        self.refrac_hid = refrac_hid 
        self.refrac_out = refrac_out

        self.one_spike_hid = one_spike_hid
        self.one_spike_out = one_spike_out

        self.tc_trace_input = tc_trace_input
        self.tc_trace_hid = tc_trace_hid
        self.tc_trace_out = tc_trace_out

        self.tc_decay_hid = tc_decay_hid
        self.tc_decay_out = tc_decay_out

        self.tc_plus_out = tc_plus_out
        self.tc_minus_out = tc_minus_out
        self.tc_e_trace_out = tc_e_trace_out

        input_layer = Input(
            n=self.n_inpt, shape=self.inpt_shape, traces=True, tc_trace=self.tc_trace_input
        )
        self.add_layer(input_layer, name="X")

        hidden_layer = DiehlAndCookNodes(
            n=self.n_neurons_hid,
            traces=True,
            rest=-65.0,
            reset=-65.0, # default: -60
            thresh=thresh_hid, # default: -52
            refrac=self.refrac_hid,
            tc_decay=self.tc_decay_hid,
            tc_trace=self.tc_trace_hid,
            theta_plus=0, # default: theta_plus
            tc_theta_decay=tc_theta_decay,
            one_spike=self.one_spike_hid,
        )

        self.add_layer(hidden_layer, name="H")

        w = self.w_init_hid * torch.rand(self.n_inpt, self.n_neurons_hid)
        input_connection = Connection(
            source=self.layers["X"],
            target=self.layers["H"],
            w=w,
            update_rule=PostPre,
            nu=nu_hid,
            reduction=reduction,
            wmin=wmin,
            wmax=wmax,
            norm=norm,
        )
        self.add_connection(input_connection, source="X", target="H")

        loaded_net = load(hidden_layer_path)
        self.connections['X', 'H'].update_rule = NoOp(
            connection=loaded_net.connections['X', 'Y'], nu=loaded_net.connections['X', 'Y'].nu
        )
        self.connections['X', 'H'].w = loaded_net.connections['X', 'Y'].w
        self.layers['H'].tc_theta_decay = torch.tensor(0.0, dtype=torch.int8)
        self.layers['H'].theta_plus = torch.tensor(0.0, dtype=torch.int8)

        w = -self.inh_hid * (
            torch.ones(self.n_neurons_hid, self.n_neurons_hid)
            - torch.diag(torch.ones(self.n_neurons_hid))
        )
        recurrent_connection = Connection(
            source=self.layers["H"],
            target=self.layers["H"],
            w=w,
            wmin=-self.inh_hid,
            wmax=0,
        )
        self.add_connection(recurrent_connection, source="H", target="H")

        output_layer = DiehlAndCookNodes(
            n=self.n_neurons_out,
            traces=True,
            rest=-65.0,
            reset=-65.0, # default: -60
            thresh=thresh_out, # default: -52
            refrac=self.refrac_out,
            tc_decay=self.tc_decay_out,
            tc_trace=self.tc_trace_out,
            theta_plus=0, # default: theta_plus
            tc_theta_decay=tc_theta_decay,
            one_spike=self.one_spike_out,
        )
        self.add_layer(output_layer, name="Y")

        w = self.w_init_out * torch.rand(self.n_neurons_hid, self.n_neurons_out)
        # w = torch.normal(
        #     self.w_init_out,
        #     0.1, 
        #     size=(self.n_neurons_hid, self.n_neurons_out)
        # )
        output_connection = Connection(
            source=self.layers["H"],
            target=self.layers["Y"],
            w=w,
            update_rule=MSTDPET,
            nu=nu_out,
            reduction=reduction,
            wmin=wmin,
            wmax=wmax,
            norm=norm,
            tc_plus=self.tc_plus_out,
            tc_minus=self.tc_minus_out,
            tc_e_trace=self.tc_e_trace_out,
        )
        self.add_connection(output_connection, source="H", target="Y")

        w = -self.inh_out * (
            torch.ones(self.n_neurons_out, self.n_neurons_out)
            - torch.diag(torch.ones(self.n_neurons_out))
        )
        recurrent_connection_out = Connection(
            source=self.layers["Y"],
            target=self.layers["Y"],
            w=w,
            wmin=-self.inh_out,
            wmax=0,
        )
        self.add_connection(recurrent_connection_out, source="Y", target="Y")


class Diehl2LayersWTAOutput(Network):
    # language=rst
    """
    Slightly modifies the spiking neural network architecture from `(Diehl & Cook 2015)
    <https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full>`_ by removing
    the inhibitory layer and replacing it with a recurrent inhibitory connection in the
    output layer (what used to be the excitatory layer).
    """

    def __init__(
        self,
        n_inpt: int = 60,
        n_neurons_hid: int = 100,
        n_neurons_out: int = 2,
        inh_hid: float = 0.1,
        inh_out: float = 17.5,
        dt: float = 1.0,
        nu_hid: Optional[Union[float, Sequence[float]]] = (1e-4, 1e-2),
        nu_out: Optional[Union[float, Sequence[float]]] = (1e-4, 1e-2),
        reduction: Optional[callable] = None,
        wmin: Optional[float] = 0.0,
        wmax: Optional[float] = 1.0,
        norm: float = None,
        theta_plus: float = 0.0,
        tc_theta_decay: float = 1e7,
        inpt_shape: Optional[Iterable[int]] = None,
        hidden_layer_path: str = None,
        w_init_hid=0.5,
        w_init_out=0.5,
        thresh_hid=-58.0,
        thresh_out=-61.0,
        refrac_hid=5,
        refrac_out=1,
        one_spike_hid=False,
        one_spike_out=False,
        tc_trace_input=20.0,
        tc_decay_hid=15.0,
        tc_decay_out=15.0,
        tc_trace_hid=10.0,
        tc_trace_out=10.0,
        tc_plus_out=10.0,
        tc_minus_out=10.0,
        tc_e_trace_out=15.0,
        wmin_out: Optional[float] = 0.0,
        wmax_out: Optional[float] = 1.0,
    ) -> None:
        # language=rst
        """
        Constructor for class ``DiehlAndCook2015v2``.

        :param n_inpt: Number of input neurons. Matches the 1D size of the input data.
        :param n_neurons: Number of excitatory, inhibitory neurons.
        :param inh: Strength of synapse weights from inhibitory to excitatory layer.
        :param dt: Simulation time step.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events,
            respectively.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param wmin: Minimum allowed weight on input to excitatory synapses.
        :param wmax: Maximum allowed weight on input to excitatory synapses.
        :param norm: Input to excitatory layer connection weights normalization
            constant.
        :param theta_plus: On-spike increment of ``DiehlAndCookNodes`` membrane
            threshold potential.
        :param tc_theta_decay: Time constant of ``DiehlAndCookNodes`` threshold
            potential decay.
        :param inpt_shape: The dimensionality of the input layer.
        """
        super().__init__(dt=dt)

        self.n_inpt = n_inpt
        self.inpt_shape = inpt_shape
        self.n_neurons_hid = n_neurons_hid
        self.n_neurons_out = n_neurons_out
        self.inh_hid = inh_hid
        self.inh_out = inh_out
        self.dt = dt

        self.w_init_hid = w_init_hid
        self.w_init_out = w_init_out

        self.wmin_out = wmin_out
        self.wmax_out = wmax_out
        
        self.refrac_hid = refrac_hid 
        self.refrac_out = refrac_out

        self.one_spike_hid = one_spike_hid
        self.one_spike_out = one_spike_out

        self.tc_trace_input = tc_trace_input
        self.tc_trace_hid = tc_trace_hid
        self.tc_trace_out = tc_trace_out

        self.tc_decay_hid = tc_decay_hid
        self.tc_decay_out = tc_decay_out

        self.tc_plus_out = tc_plus_out
        self.tc_minus_out = tc_minus_out
        self.tc_e_trace_out = tc_e_trace_out

        input_layer = Input(
            n=self.n_inpt, shape=self.inpt_shape, traces=True, tc_trace=self.tc_trace_input
        )
        self.add_layer(input_layer, name="X")

        hidden_layer = DiehlAndCookNodes(
            n=self.n_neurons_hid,
            traces=True,
            rest=-65.0,
            reset=-65.0, # default: -60
            thresh=thresh_hid, # default: -52
            refrac=self.refrac_hid,
            tc_decay=self.tc_decay_hid,
            tc_trace=self.tc_trace_hid,
            theta_plus=0, # default: theta_plus
            tc_theta_decay=tc_theta_decay,
            one_spike=self.one_spike_hid,
        )

        self.add_layer(hidden_layer, name="H")

        w = self.w_init_hid * torch.rand(self.n_inpt, self.n_neurons_hid)
        input_connection = Connection(
            source=self.layers["X"],
            target=self.layers["H"],
            w=w,
            update_rule=PostPre,
            nu=nu_hid,
            reduction=reduction,
            wmin=wmin,
            wmax=wmax,
            norm=norm,
        )
        self.add_connection(input_connection, source="X", target="H")

        loaded_net = load(hidden_layer_path)
        self.connections['X', 'H'].update_rule = NoOp(
            connection=loaded_net.connections['X', 'Y'], nu=loaded_net.connections['X', 'Y'].nu
        )
        self.connections['X', 'H'].w = loaded_net.connections['X', 'Y'].w
        self.layers['H'].tc_theta_decay = torch.tensor(0.0, dtype=torch.int8)
        self.layers['H'].theta_plus = torch.tensor(0.0, dtype=torch.int8)

        w = -self.inh_hid * (
            torch.ones(self.n_neurons_hid, self.n_neurons_hid)
            - torch.diag(torch.ones(self.n_neurons_hid))
        )
        recurrent_connection = Connection(
            source=self.layers["H"],
            target=self.layers["H"],
            w=w,
            wmin=-self.inh_hid,
            wmax=0,
        )
        self.add_connection(recurrent_connection, source="H", target="H")

        output_layer = DiehlAndCookNodesModified(
            n=self.n_neurons_out,
            traces=True,
            rest=-65.0,
            reset=-65.0, # default: -60
            thresh=thresh_out, # default: -52
            refrac=self.refrac_out,
            tc_decay=self.tc_decay_out,
            tc_trace=self.tc_trace_out,
            theta_plus=0, # default: theta_plus
            tc_theta_decay=tc_theta_decay,
            one_spike=self.one_spike_out,
        )
        self.add_layer(output_layer, name="Y")

        w = self.w_init_out * torch.rand(self.n_neurons_hid, self.n_neurons_out)
        # w = torch.normal(
        #     self.w_init_out,
        #     0.1, 
        #     size=(self.n_neurons_hid, self.n_neurons_out)
        # )
        output_connection = Connection(
            source=self.layers["H"],
            target=self.layers["Y"],
            w=w,
            update_rule=MSTDPET,
            nu=nu_out,
            reduction=reduction,
            wmin=self.wmin_out,
            wmax=self.wmax_out,
            norm=norm,
            tc_plus=self.tc_plus_out,
            tc_minus=self.tc_minus_out,
            tc_e_trace=self.tc_e_trace_out,
        )
        self.add_connection(output_connection, source="H", target="Y")

        w = -self.inh_out * (
            torch.ones(self.n_neurons_out, self.n_neurons_out)
            - torch.diag(torch.ones(self.n_neurons_out))
        )
        recurrent_connection_out = Connection(
            source=self.layers["Y"],
            target=self.layers["Y"],
            w=w,
            wmin=-self.inh_out,
            wmax=0,
        )
        self.add_connection(recurrent_connection_out, source="Y", target="Y")

    def run(
        self, inputs, time: int, one_step=False, learning=True, output_layer='OUT',
        ext_reward=-1, actor_network=None, obs=None, 
        mean_reward=None, step=0, **kwargs
    ) -> None:
        # language=rst
        """
        Simulate network for given inputs and time.

        :param inputs: Dictionary of ``Tensor``s of shape ``[time, *input_shape]`` or
                      ``[time, batch_size, *input_shape]``.
        :param time: Simulation time.
        :param one_step: Whether to run the network in "feed-forward" mode, where inputs
            propagate all the way through the network in a single simulation time step.
            Layers are updated in the order they are added to the network.

        Keyword arguments:

        :param Dict[str, torch.Tensor] clamp: Mapping of layer names to boolean masks if
            neurons should be clamped to spiking. The ``Tensor``s have shape
            ``[n_neurons]`` or ``[time, n_neurons]``.
        :param Dict[str, torch.Tensor] unclamp: Mapping of layer names to boolean masks
            if neurons should be clamped to not spiking. The ``Tensor``s should have
            shape ``[n_neurons]`` or ``[time, n_neurons]``.
        :param Dict[str, torch.Tensor] injects_v: Mapping of layer names to boolean
            masks if neurons should be added voltage. The ``Tensor``s should have shape
            ``[n_neurons]`` or ``[time, n_neurons]``.
        :param Union[float, torch.Tensor] reward: Scalar value used in reward-modulated
            learning.
        :param Dict[Tuple[str], torch.Tensor] masks: Mapping of connection names to
            boolean masks determining which weights to clamp to zero.

        **Example:**

        .. code-block:: python

            import torch
            import matplotlib.pyplot as plt

            from bindsnet.network import Network
            from bindsnet.network.nodes import Input
            from bindsnet.network.monitors import Monitor

            # Build simple network.
            network = Network()
            network.add_layer(Input(500), name='I')
            network.add_monitor(Monitor(network.layers['I'], state_vars=['s']), 'I')

            # Generate spikes by running Bernoulli trials on Uniform(0, 0.5) samples.
            spikes = torch.bernoulli(0.5 * torch.rand(500, 500))

            # Run network simulation.
            network.run(inputs={'I' : spikes}, time=500)

            # Look at input spiking activity.
            spikes = network.monitors['I'].get('s')
            plt.matshow(spikes, cmap='binary')
            plt.xticks(()); plt.yticks(());
            plt.xlabel('Time'); plt.ylabel('Neuron index')
            plt.title('Input spiking')
            plt.show()
        """
        # Parse keyword arguments.
        clamps = kwargs.get("clamp", {})
        unclamps = kwargs.get("unclamp", {})
        masks = kwargs.get("masks", {})
        injects_v = kwargs.get("injects_v", {})

        # Compute reward.
        
        # reward = kwargs["reward"]

        self.learning = learning
        
        # Dynamic setting of batch size.
        if inputs != {}:
            for key in inputs:
                # goal shape is [time, batch, n_0, ...]
                if len(inputs[key].size()) == 1:
                    # current shape is [n_0, ...]
                    # unsqueeze twice to make [1, 1, n_0, ...]
                    inputs[key] = inputs[key].unsqueeze(0).unsqueeze(0)
                elif len(inputs[key].size()) == 2:
                    # current shape is [time, n_0, ...]
                    # unsqueeze dim 1 so that we have
                    # [time, 1, n_0, ...]
                    inputs[key] = inputs[key].unsqueeze(1)

            for key in inputs:
                # batch dimension is 1, grab this and use for batch size
                if inputs[key].size(1) != self.batch_size:
                    self.batch_size = inputs[key].size(1)

                    for l in self.layers:
                        self.layers[l].set_batch_size(self.batch_size)

                    for m in self.monitors:
                        self.monitors[m].reset_state_variables()

                break

        # Effective number of timesteps.
        timesteps = int(time / self.dt)

        # Simulate network activity for `time` timesteps.
        for t in range(timesteps):
            # Get input to all layers (synchronous mode).
            current_inputs = {}
            if not one_step:
                current_inputs.update(self._get_inputs())

            for l in self.layers:
                # Update each layer of nodes.
                if l in inputs:
                    if l in current_inputs:
                        current_inputs[l] += inputs[l][t]
                    else:
                        current_inputs[l] = inputs[l][t]

                if one_step:
                    # Get input to this layer (one-step mode).
                    current_inputs.update(self._get_inputs(layers=[l]))

                self.layers[l].forward(x=current_inputs[l])


                # Clamp neurons to spike.
                clamp = clamps.get(l, None)
                if clamp is not None:
                    if clamp.ndimension() == 1:
                        self.layers[l].s[:, clamp] = 1
                    else:
                        self.layers[l].s[:, clamp[t]] = 1

                # Clamp neurons not to spike.
                unclamp = unclamps.get(l, None)
                if unclamp is not None:
                    if unclamp.ndimension() == 1:
                        self.layers[l].s[unclamp] = 0
                    else:
                        self.layers[l].s[unclamp[t]] = 0

                # Inject voltage to neurons.
                inject_v = injects_v.get(l, None)
                if inject_v is not None:
                    if inject_v.ndimension() == 1:
                        self.layers[l].v += inject_v
                    else:
                        self.layers[l].v += inject_v[t]


            if self.reward_fn is not None:
                # Call reward function, passing self (Network class) and t (time step)
                kwargs["reward"] = self.reward_fn.compute(self, t, **kwargs)
                
            # Run synapse updates.
            if self.learning:
                for c in self.connections:
                    self.connections[c].update(
                        mask=masks.get(c, None), learning=self.learning, step=step, **kwargs
                    )

            # Get input to all layers.
            if not one_step:
                current_inputs.update(self._get_inputs())

            # Record state variables of interest.
            for m in self.monitors:
                self.monitors[m].record()

        # Re-normalize connections.
        for c in self.connections:
            self.connections[c].normalize()


class Diehl2LayersGroupWTAOutput(Diehl2LayersWTAOutput):
    def __init__(self,
                 n_actions: int = 2,
                 group_size: int = 10,
                 n_inpt: int = 60,
                 n_neurons_hid: int = 1000,
                 inh_hid: float = .1,
                 inh_out: float = .1,
                 dt: float = .5,
                 nu_hid=(2.5e-6, 1e-5),
                 nu_out=(4e-3, 1e-2),
                 hidden_layer_path: str = None,
                 **kwargs):
        self.n_actions   = n_actions
        self.group_size  = group_size
        n_neurons_out    = n_actions * group_size

        super().__init__(n_inpt=n_inpt,
                         n_neurons_hid=n_neurons_hid,
                         n_neurons_out=n_neurons_out,
                         inh_hid=inh_hid,
                         inh_out=inh_out,
                         dt=dt,
                         nu_hid=nu_hid,
                         nu_out=nu_out,
                         hidden_layer_path=hidden_layer_path,
                         one_spike_out=True,   
                         **kwargs)

    def get_action(self, spike_counts: torch.Tensor) -> int:

        sc = spike_counts.view(self.n_actions, self.group_size).sum(1)
        return torch.argmax(sc).item()


class Diehl2LayersWTAOutputFBSTDP(Network):
    # language=rst
    """
    Slightly modifies the spiking neural network architecture from `(Diehl & Cook 2015)
    <https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full>`_ by removing
    the inhibitory layer and replacing it with a recurrent inhibitory connection in the
    output layer (what used to be the excitatory layer).
    """

    def __init__(
        self,
        n_actions: int = 2,
        group_size: int = 10,
        n_inpt: int = 60,
        n_neurons_hid: int = 100,
        n_neurons_out: int = 2,
        inh_hid: float = 0.1,
        inh_out: float = 17.5,
        dt: float = 1.0,
        nu_hid: Optional[Union[float, Sequence[float]]] = (1e-4, 1e-2),
        nu_out: Optional[Union[float, Sequence[float]]] = (1e-4, 1e-2),
        reduction: Optional[callable] = None,
        wmin: Optional[float] = 0.0,
        wmax: Optional[float] = 1.0,
        norm: float = None,
        theta_plus: float = 0.05,
        tc_theta_decay: float = 1e7,
        inpt_shape: Optional[Iterable[int]] = None,
        hidden_layer_path: str = None,
        w_init_hid=0.5,
        w_init_out=0.5,
        thresh_hid=-58.0,
        thresh_out=-61.0,
        refrac_hid=5,
        refrac_out=1,
        one_spike_hid=False,
        one_spike_out=False,
        tc_trace_input=20.0,
        tc_decay_hid=15.0,
        tc_decay_out=15.0,
        tc_trace_hid=10.0,
        tc_trace_out=10.0,
        tc_plus_out=10.0,
        tc_minus_out=10.0,
        tc_e_trace_out=15.0,
        num_actions=2,
    ) -> None:
        # language=rst
        """
        Constructor for class ``DiehlAndCook2015v2``.

        :param n_inpt: Number of input neurons. Matches the 1D size of the input data.
        :param n_neurons: Number of excitatory, inhibitory neurons.
        :param inh: Strength of synapse weights from inhibitory to excitatory layer.
        :param dt: Simulation time step.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events,
            respectively.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param wmin: Minimum allowed weight on input to excitatory synapses.
        :param wmax: Maximum allowed weight on input to excitatory synapses.
        :param norm: Input to excitatory layer connection weights normalization
            constant.
        :param theta_plus: On-spike increment of ``DiehlAndCookNodes`` membrane
            threshold potential.
        :param tc_theta_decay: Time constant of ``DiehlAndCookNodes`` threshold
            potential decay.
        :param inpt_shape: The dimensionality of the input layer.
        """
        super().__init__(dt=dt)

        self.n_actions   = n_actions
        self.group_size  = group_size
        n_neurons_out    = n_actions * group_size

        self.n_inpt = n_inpt
        self.inpt_shape = inpt_shape
        self.n_neurons_hid = n_neurons_hid
        self.n_neurons_out = n_neurons_out
        self.inh_hid = inh_hid
        self.inh_out = inh_out
        self.dt = dt

        self.w_init_hid = w_init_hid
        self.w_init_out = w_init_out
        
        self.refrac_hid = refrac_hid 
        self.refrac_out = refrac_out

        self.one_spike_hid = one_spike_hid
        self.one_spike_out = one_spike_out

        self.tc_trace_input = tc_trace_input
        self.tc_trace_hid = tc_trace_hid
        self.tc_trace_out = tc_trace_out

        self.tc_decay_hid = tc_decay_hid
        self.tc_decay_out = tc_decay_out

        self.tc_plus_out = tc_plus_out
        self.tc_minus_out = tc_minus_out
        self.tc_e_trace_out = tc_e_trace_out

        input_layer = Input(
            n=self.n_inpt, shape=self.inpt_shape, traces=True, tc_trace=self.tc_trace_input
        )
        self.add_layer(input_layer, name="X")

        hidden_layer = DiehlAndCookNodes(
            n=self.n_neurons_hid,
            traces=True,
            rest=-65.0,
            reset=-65.0, # default: -60
            thresh=thresh_hid, # default: -52
            refrac=self.refrac_hid,
            tc_decay=self.tc_decay_hid,
            tc_trace=self.tc_trace_hid,
            theta_plus=0, # default: theta_plus
            tc_theta_decay=tc_theta_decay,
            one_spike=self.one_spike_hid,
        )

        self.add_layer(hidden_layer, name="H")

        w = self.w_init_hid * torch.rand(self.n_inpt, self.n_neurons_hid)
        input_connection = Connection(
            source=self.layers["X"],
            target=self.layers["H"],
            w=w,
            update_rule=PostPre,
            nu=nu_hid,
            reduction=reduction,
            wmin=wmin,
            wmax=wmax,
            norm=norm,
        )
        self.add_connection(input_connection, source="X", target="H")

        loaded_net = load(hidden_layer_path)
        self.connections['X', 'H'].update_rule = NoOp(
            connection=loaded_net.connections['X', 'Y'], nu=loaded_net.connections['X', 'Y'].nu
        )
        self.connections['X', 'H'].w = loaded_net.connections['X', 'Y'].w
        self.layers['H'].tc_theta_decay = torch.tensor(0.0, dtype=torch.int8)
        self.layers['H'].theta_plus = torch.tensor(0.0, dtype=torch.int8)

        w = -self.inh_hid * (
            torch.ones(self.n_neurons_hid, self.n_neurons_hid)
            - torch.diag(torch.ones(self.n_neurons_hid))
        )
        recurrent_connection = Connection(
            source=self.layers["H"],
            target=self.layers["H"],
            w=w,
            wmin=-self.inh_hid,
            wmax=0,
        )
        self.add_connection(recurrent_connection, source="H", target="H")

        output_layer = DiehlAndCookNodesModified(
            n=self.n_neurons_out,
            traces=True,
            rest=-65.0,
            reset=-65.0, # default: -60
            thresh=thresh_out, # default: -52
            refrac=self.refrac_out,
            tc_decay=self.tc_decay_out,
            tc_trace=self.tc_trace_out,
            theta_plus=0, # default: theta_plus
            tc_theta_decay=tc_theta_decay,
            one_spike=self.one_spike_out,
        )
        self.add_layer(output_layer, name="Y")

        w = self.w_init_out * torch.rand(self.n_neurons_hid, self.n_neurons_out)
        # w = torch.normal(
        #     self.w_init_out,
        #     0.1, 
        #     size=(self.n_neurons_hid, self.n_neurons_out)
        # )
        output_connection = Connection(
            source=self.layers["H"],
            target=self.layers["Y"],
            w=w,
            update_rule=FBTDSTDP,
            nu=nu_out,
            reduction=reduction,
            wmin=wmin,
            wmax=wmax,
            norm=norm,
            fb_gate=False,
            tc_a_trace=20.0,
            num_actions=num_actions,
            tc_plus=self.tc_plus_out,
            tc_minus=self.tc_minus_out,
            tc_e_trace=self.tc_e_trace_out,
        )
        self.add_connection(output_connection, source="H", target="Y")

        w = -self.inh_out * (
            torch.ones(self.n_neurons_out, self.n_neurons_out)
            - torch.diag(torch.ones(self.n_neurons_out))
        )
        recurrent_connection_out = Connection(
            source=self.layers["Y"],
            target=self.layers["Y"],
            w=w,
            wmin=-self.inh_out,
            wmax=0,
        )
        self.add_connection(recurrent_connection_out, source="Y", target="Y")

    def run(
        self, inputs, time: int, one_step=False, learning=True, output_layer='OUT',
        ext_reward=-1, actor_network=None, obs=None, 
        mean_reward=None, step=0, **kwargs
    ) -> None:
        # language=rst
        """
        Simulate network for given inputs and time.

        :param inputs: Dictionary of ``Tensor``s of shape ``[time, *input_shape]`` or
                      ``[time, batch_size, *input_shape]``.
        :param time: Simulation time.
        :param one_step: Whether to run the network in "feed-forward" mode, where inputs
            propagate all the way through the network in a single simulation time step.
            Layers are updated in the order they are added to the network.

        Keyword arguments:

        :param Dict[str, torch.Tensor] clamp: Mapping of layer names to boolean masks if
            neurons should be clamped to spiking. The ``Tensor``s have shape
            ``[n_neurons]`` or ``[time, n_neurons]``.
        :param Dict[str, torch.Tensor] unclamp: Mapping of layer names to boolean masks
            if neurons should be clamped to not spiking. The ``Tensor``s should have
            shape ``[n_neurons]`` or ``[time, n_neurons]``.
        :param Dict[str, torch.Tensor] injects_v: Mapping of layer names to boolean
            masks if neurons should be added voltage. The ``Tensor``s should have shape
            ``[n_neurons]`` or ``[time, n_neurons]``.
        :param Union[float, torch.Tensor] reward: Scalar value used in reward-modulated
            learning.
        :param Dict[Tuple[str], torch.Tensor] masks: Mapping of connection names to
            boolean masks determining which weights to clamp to zero.

        **Example:**

        .. code-block:: python

            import torch
            import matplotlib.pyplot as plt

            from bindsnet.network import Network
            from bindsnet.network.nodes import Input
            from bindsnet.network.monitors import Monitor

            # Build simple network.
            network = Network()
            network.add_layer(Input(500), name='I')
            network.add_monitor(Monitor(network.layers['I'], state_vars=['s']), 'I')

            # Generate spikes by running Bernoulli trials on Uniform(0, 0.5) samples.
            spikes = torch.bernoulli(0.5 * torch.rand(500, 500))

            # Run network simulation.
            network.run(inputs={'I' : spikes}, time=500)

            # Look at input spiking activity.
            spikes = network.monitors['I'].get('s')
            plt.matshow(spikes, cmap='binary')
            plt.xticks(()); plt.yticks(());
            plt.xlabel('Time'); plt.ylabel('Neuron index')
            plt.title('Input spiking')
            plt.show()
        """
        # Parse keyword arguments.
        clamps = kwargs.get("clamp", {})
        unclamps = kwargs.get("unclamp", {})
        masks = kwargs.get("masks", {})
        injects_v = kwargs.get("injects_v", {})

        # Compute reward.
        
        # reward = kwargs["reward"]

        self.learning = learning
        
        # Dynamic setting of batch size.
        if inputs != {}:
            for key in inputs:
                # goal shape is [time, batch, n_0, ...]
                if len(inputs[key].size()) == 1:
                    # current shape is [n_0, ...]
                    # unsqueeze twice to make [1, 1, n_0, ...]
                    inputs[key] = inputs[key].unsqueeze(0).unsqueeze(0)
                elif len(inputs[key].size()) == 2:
                    # current shape is [time, n_0, ...]
                    # unsqueeze dim 1 so that we have
                    # [time, 1, n_0, ...]
                    inputs[key] = inputs[key].unsqueeze(1)

            for key in inputs:
                # batch dimension is 1, grab this and use for batch size
                if inputs[key].size(1) != self.batch_size:
                    self.batch_size = inputs[key].size(1)

                    for l in self.layers:
                        self.layers[l].set_batch_size(self.batch_size)

                    for m in self.monitors:
                        self.monitors[m].reset_state_variables()

                break

        # Effective number of timesteps.
        timesteps = int(time / self.dt)

        # Simulate network activity for `time` timesteps.
        for t in range(timesteps):
            # Get input to all layers (synchronous mode).
            current_inputs = {}
            if not one_step:
                current_inputs.update(self._get_inputs())

            for l in self.layers:
                # Update each layer of nodes.
                if l in inputs:
                    if l in current_inputs:
                        current_inputs[l] += inputs[l][t]
                    else:
                        current_inputs[l] = inputs[l][t]

                if one_step:
                    # Get input to this layer (one-step mode).
                    current_inputs.update(self._get_inputs(layers=[l]))

                self.layers[l].forward(x=current_inputs[l])


                # Clamp neurons to spike.
                clamp = clamps.get(l, None)
                if clamp is not None:
                    if clamp.ndimension() == 1:
                        self.layers[l].s[:, clamp] = 1
                    else:
                        self.layers[l].s[:, clamp[t]] = 1

                # Clamp neurons not to spike.
                unclamp = unclamps.get(l, None)
                if unclamp is not None:
                    if unclamp.ndimension() == 1:
                        self.layers[l].s[unclamp] = 0
                    else:
                        self.layers[l].s[unclamp[t]] = 0

                # Inject voltage to neurons.
                inject_v = injects_v.get(l, None)
                if inject_v is not None:
                    if inject_v.ndimension() == 1:
                        self.layers[l].v += inject_v
                    else:
                        self.layers[l].v += inject_v[t]


            if self.reward_fn is not None:
                # Call reward function, passing self (Network class) and t (time step)
                kwargs["reward"] = self.reward_fn.compute(self, t, **kwargs)
                
            # Run synapse updates.
            if self.learning:
                for c in self.connections:
                    self.connections[c].update(
                        mask=masks.get(c, None), learning=self.learning, step=step, **kwargs
                    )

            # Get input to all layers.
            if not one_step:
                current_inputs.update(self._get_inputs())

            # Record state variables of interest.
            for m in self.monitors:
                self.monitors[m].record()

        # Re-normalize connections.
        for c in self.connections:
            self.connections[c].normalize()

    def get_action(self, spike_counts: torch.Tensor) -> int:
        sc = spike_counts.view(self.n_actions, self.group_size).sum(1)
        return torch.argmax(sc).item()
    

class Diehl1LayerFourierAC(Network):
    # language=rst
    """
    1 layer actor and critic for fourier encoded input
    """

    def __init__(
        self,
        k_inpt=5,
        n_input_neurons=1296,
        n_neurons_out_actor: int = 2,
        n_neurons_out_critic: int = 100,
        inh_out_actor: float = 0.1,
        dt: float = 1.0,
        nu_out_actor: Optional[Union[float, Sequence[float]]] = (1e-4, 1e-2),
        nu_out_critic: Optional[Union[float, Sequence[float]]] = (1e-4, 1e-2),
        reduction: Optional[callable] = None,
        wmin_actor: Optional[float] = 0.0,
        wmax_actor: Optional[float] = 1.0,
        wmin_critic: Optional[float] = 0.0,
        wmax_critic: Optional[float] = 1.0,
        norm: float = None,
        theta_plus_actor: float = 0.0,
        tc_theta_decay_actor: float = 1e7,
        theta_plus_critic: float = 0.0,
        tc_theta_decay_critic: float = 1e7,
        w_init_out_actor=0.5,
        w_init_out_critic=0.5,
        thresh_out_actor=-61.0,
        thresh_out_critic=-61.0,
        refrac_out_actor=0,
        refrac_out_critic=0,
        one_spike_out_actor=False,
        one_spike_out_critic=False,
        tc_trace_input=20.0,
        tc_decay_out_actor=15.0,
        tc_decay_out_critic=15.0,
        tc_trace_out_actor=10.0,
        tc_trace_out_critic=10.0,
        tc_plus_out_actor=10.0,
        tc_plus_out_critic=10.0,
        tc_minus_out_actor=10.0,
        tc_minus_out_critic=10.0,
        tc_e_trace_out_actor=15.0,
        tc_e_trace_out_critic=15.0,
    ) -> None:
        super().__init__(dt=dt)

        self.k_inpt = k_inpt
        self.n_input_neurons = n_input_neurons
        self.n_neurons_out_actor = n_neurons_out_actor
        self.n_neurons_out_critic = n_neurons_out_critic
        self.inh_out_actor = inh_out_actor
        self.dt = dt

        self.w_init_out_actor = w_init_out_actor
        self.w_init_out_critic = w_init_out_critic
        
        self.refrac_out_actor = refrac_out_actor
        self.refrac_out_critic = refrac_out_critic

        self.one_spike_out_actor = one_spike_out_actor
        self.one_spike_out_critic = one_spike_out_critic

        self.nu_out_actor = nu_out_actor
        self.nu_out_critic = nu_out_critic

        self.theta_plus_actor = theta_plus_actor
        self.theta_plus_critic = theta_plus_critic 

        self.tc_theta_decay_actor = tc_theta_decay_actor
        self.tc_theta_decay_critic = tc_theta_decay_critic

        self.wmin_actor = wmin_actor
        self.wmin_critic = wmin_critic

        self.wmax_actor = wmax_actor
        self.wmax_critic = wmax_critic

        self.tc_trace_input = tc_trace_input
        self.tc_trace_out_actor = tc_trace_out_actor
        self.tc_trace_out_critic = tc_trace_out_critic

        self.tc_decay_out_actor = tc_decay_out_actor
        self.tc_decay_out_critic = tc_decay_out_critic

        self.tc_plus_out_actor = tc_plus_out_actor
        self.tc_plus_out_critic = tc_plus_out_critic
        self.tc_minus_out_actor = tc_minus_out_actor
        self.tc_minus_out_critic = tc_minus_out_critic
        self.tc_e_trace_out_actor = tc_e_trace_out_actor
        self.tc_e_trace_out_critic = tc_e_trace_out_critic

        self.thresh_out_actor = thresh_out_actor
        self.thresh_out_critic = thresh_out_critic

        input_layer = Input(n=self.n_input_neurons, traces=True, tc_trace=self.tc_trace_input)
        self.add_layer(input_layer, name="X")

        value_neuron = DiehlAndCookNodesModified(
            n=self.n_neurons_out_critic, traces=True,
            traces_additive=True,
            rest=-65.0,
            reset=-65.0,
            thresh=self.thresh_out_critic,        
            tc_decay=self.tc_decay_out_critic,          # ms  (integrator)
            tc_trace=self.tc_trace_out_critic,           # post-syn trace
            theta_plus=self.theta_plus_critic,
            tc_theta_decay=self.tc_theta_decay_critic,
            refrac=self.refrac_out_critic,
            one_spike=self.one_spike_out_critic
        )
    
        self.add_layer(value_neuron, name='V')

        w_critic = self.w_init_out_critic * torch.rand(self.n_input_neurons, self.n_neurons_out_critic)

        critic_connection = Connection(
            source=self.layers["X"],
            target=self.layers["V"],
            w=w_critic,
            update_rule=MSTDPET,
            nu=self.nu_out_critic,
            reduction=reduction,
            wmin=self.wmin_critic,
            wmax=self.wmax_critic,
            norm=norm,
        )
        self.add_connection(critic_connection, source="X", target="V")

        actor_output_layer = DiehlAndCookNodesModified(
            n=self.n_neurons_out_actor,
            traces=True,
            rest=-65.0,
            reset=-65.0, # default: -60
            thresh=self.thresh_out_actor, # default: -52
            refrac=self.refrac_out_actor,
            tc_decay=self.tc_decay_out_actor,
            tc_trace=self.tc_trace_out_actor,
            theta_plus=self.theta_plus_actor,
            tc_theta_decay=self.tc_theta_decay_actor,
            one_spike=self.one_spike_out_actor,
        )
        self.add_layer(actor_output_layer, name="Y")

        w_actor = self.w_init_out_actor * torch.rand(self.n_input_neurons, self.n_neurons_out_actor)
        actor_output_connection = Connection(
            source=self.layers["X"],
            target=self.layers["Y"],
            w=w_actor,
            update_rule=MSTDPET,
            nu=self.nu_out_actor,
            reduction=reduction,
            wmin=self.wmin_actor,
            wmax=self.wmax_actor,
            norm=norm,
            tc_plus=self.tc_plus_out_actor,
            tc_minus=self.tc_minus_out_actor,
            tc_e_trace=self.tc_e_trace_out_actor,
        )
        self.add_connection(actor_output_connection, source="X", target="Y")

        w_inh_actor = -self.inh_out_actor * (
            torch.ones(self.n_neurons_out_actor, self.n_neurons_out_actor)
            - torch.diag(torch.ones(self.n_neurons_out_actor))
        )
        recurrent_connection_out = Connection(
            source=self.layers["Y"],
            target=self.layers["Y"],
            w=w_inh_actor,
            wmin=-self.inh_out_actor,
            wmax=0,
        )
        self.add_connection(recurrent_connection_out, source="Y", target="Y")

    # def run(
    #     self, inputs, time: int, one_step=False, learning=True, output_layer='OUT',
    #     ext_reward=-1, actor_network=None, obs=None, 
    #     mean_reward=None, step=0, **kwargs
    # ) -> None:
    #     # Parse keyword arguments.
    #     clamps = kwargs.get("clamp", {})
    #     unclamps = kwargs.get("unclamp", {})
    #     masks = kwargs.get("masks", {})
    #     injects_v = kwargs.get("injects_v", {})

    #     # Compute reward.
        
    #     # reward = kwargs["reward"]

    #     self.learning = learning
        
    #     # Dynamic setting of batch size.
    #     if inputs != {}:
    #         for key in inputs:
    #             # goal shape is [time, batch, n_0, ...]
    #             if len(inputs[key].size()) == 1:
    #                 # current shape is [n_0, ...]
    #                 # unsqueeze twice to make [1, 1, n_0, ...]
    #                 inputs[key] = inputs[key].unsqueeze(0).unsqueeze(0)
    #             elif len(inputs[key].size()) == 2:
    #                 # current shape is [time, n_0, ...]
    #                 # unsqueeze dim 1 so that we have
    #                 # [time, 1, n_0, ...]
    #                 inputs[key] = inputs[key].unsqueeze(1)

    #         for key in inputs:
    #             # batch dimension is 1, grab this and use for batch size
    #             if inputs[key].size(1) != self.batch_size:
    #                 self.batch_size = inputs[key].size(1)

    #                 for l in self.layers:
    #                     self.layers[l].set_batch_size(self.batch_size)

    #                 for m in self.monitors:
    #                     self.monitors[m].reset_state_variables()

    #             break

    #     # Effective number of timesteps.
    #     timesteps = int(time / self.dt)

    #     # Simulate network activity for `time` timesteps.
    #     for t in range(timesteps):
    #         # Get input to all layers (synchronous mode).
    #         current_inputs = {}
    #         if not one_step:
    #             current_inputs.update(self._get_inputs())

    #         for l in self.layers:
    #             # Update each layer of nodes.
    #             if l in inputs:
    #                 if l in current_inputs:
    #                     current_inputs[l] += inputs[l][t]
    #                 else:
    #                     current_inputs[l] = inputs[l][t]

    #             if one_step:
    #                 # Get input to this layer (one-step mode).
    #                 current_inputs.update(self._get_inputs(layers=[l]))

    #             self.layers[l].forward(x=current_inputs[l])


    #             # Clamp neurons to spike.
    #             clamp = clamps.get(l, None)
    #             if clamp is not None:
    #                 if clamp.ndimension() == 1:
    #                     self.layers[l].s[:, clamp] = 1
    #                 else:
    #                     self.layers[l].s[:, clamp[t]] = 1

    #             # Clamp neurons not to spike.
    #             unclamp = unclamps.get(l, None)
    #             if unclamp is not None:
    #                 if unclamp.ndimension() == 1:
    #                     self.layers[l].s[unclamp] = 0
    #                 else:
    #                     self.layers[l].s[unclamp[t]] = 0

    #             # Inject voltage to neurons.
    #             inject_v = injects_v.get(l, None)
    #             if inject_v is not None:
    #                 if inject_v.ndimension() == 1:
    #                     self.layers[l].v += inject_v
    #                 else:
    #                     self.layers[l].v += inject_v[t]


    #         if self.reward_fn is not None:
    #             # Call reward function, passing self (Network class) and t (time step)
    #             kwargs["reward"] = self.reward_fn.compute(self, t, **kwargs)
                
    #         # Run synapse updates.
    #         if self.learning:
    #             for c in self.connections:
    #                 self.connections[c].update(
    #                     mask=masks.get(c, None), learning=self.learning, step=step, **kwargs
    #                 )

    #         # Get input to all layers.
    #         if not one_step:
    #             current_inputs.update(self._get_inputs())

    #         # Record state variables of interest.
    #         for m in self.monitors:
    #             self.monitors[m].record()

    #     # Re-normalize connections.
    #     for c in self.connections:
    #         self.connections[c].normalize()

class Diehl1LayerGroupFourierAC(Diehl1LayerFourierAC):
    def __init__(self,
                 n_actions: int = 2,
                 group_size: int = 10,
                 **kwargs):
        self.n_actions = n_actions
        self.group_size = group_size
        kwargs["n_neurons_out_actor"] = n_actions * group_size

        super().__init__(**kwargs)

    def get_action(self, spike_counts: torch.Tensor) -> int:

        sc = spike_counts.view(self.n_actions, self.group_size).sum(1)
        return torch.argmax(sc).item()


class Diehl1LayerFourierActor(Network):
    # language=rst
    """
    1 layer actor for fourier encoded input
    """

    def __init__(
        self,
        k_inpt=5,
        n_input_neurons=1296,
        n_neurons_out_actor: int = 2,
        n_neurons_out_critic: int = 100,
        inh_out_actor: float = 0.1,
        dt: float = 1.0,
        nu_out_actor: Optional[Union[float, Sequence[float]]] = (1e-4, 1e-2),
        reduction: Optional[callable] = None,
        wmin_actor: Optional[float] = 0.0,
        wmax_actor: Optional[float] = 1.0,
        norm: float = None,
        theta_plus_actor: float = 0.0,
        tc_theta_decay_actor: float = 1e7,
        w_init_out_actor=0.5,
        thresh_out_actor=-61.0,
        refrac_out_actor=0,
        one_spike_out_actor=False,
        tc_trace_input=20.0,
        tc_decay_out_actor=15.0,
        tc_trace_out_actor=10.0,
        tc_plus_out_actor=10.0,
        tc_minus_out_actor=10.0,
        tc_e_trace_out_actor=15.0,
    ) -> None:
        super().__init__(dt=dt)

        self.k_inpt = k_inpt
        self.n_input_neurons = n_input_neurons
        self.n_neurons_out_actor = n_neurons_out_actor
        self.inh_out_actor = inh_out_actor
        self.dt = dt

        self.w_init_out_actor = w_init_out_actor
        
        self.refrac_out_actor = refrac_out_actor

        self.one_spike_out_actor = one_spike_out_actor

        self.nu_out_actor = nu_out_actor

        self.theta_plus_actor = theta_plus_actor

        self.tc_theta_decay_actor = tc_theta_decay_actor

        self.wmin_actor = wmin_actor

        self.wmax_actor = wmax_actor

        self.tc_trace_input = tc_trace_input
        self.tc_trace_out_actor = tc_trace_out_actor

        self.tc_decay_out_actor = tc_decay_out_actor

        self.tc_plus_out_actor = tc_plus_out_actor
        self.tc_minus_out_actor = tc_minus_out_actor
        self.tc_e_trace_out_actor = tc_e_trace_out_actor

        self.thresh_out_actor = thresh_out_actor

        input_layer = Input(n=self.n_input_neurons, traces=True, tc_trace=self.tc_trace_input)
        self.add_layer(input_layer, name="X")

        actor_output_layer = DiehlAndCookNodesModified(
            n=self.n_neurons_out_actor,
            traces=True,
            rest=-65.0,
            reset=-65.0, # default: -60
            thresh=self.thresh_out_actor, # default: -52
            refrac=self.refrac_out_actor,
            tc_decay=self.tc_decay_out_actor,
            tc_trace=self.tc_trace_out_actor,
            theta_plus=self.theta_plus_actor,
            tc_theta_decay=self.tc_theta_decay_actor,
            one_spike=self.one_spike_out_actor,
        )
        self.add_layer(actor_output_layer, name="Y")

        w_actor = self.w_init_out_actor * torch.rand(self.n_input_neurons, self.n_neurons_out_actor)
        actor_output_connection = Connection(
            source=self.layers["X"],
            target=self.layers["Y"],
            w=w_actor,
            update_rule=MSTDPET,
            nu=self.nu_out_actor,
            reduction=reduction,
            wmin=self.wmin_actor,
            wmax=self.wmax_actor,
            norm=norm,
            tc_plus=self.tc_plus_out_actor,
            tc_minus=self.tc_minus_out_actor,
            tc_e_trace=self.tc_e_trace_out_actor,
        )
        self.add_connection(actor_output_connection, source="X", target="Y")

        w_inh_actor = -self.inh_out_actor * (
            torch.ones(self.n_neurons_out_actor, self.n_neurons_out_actor)
            - torch.diag(torch.ones(self.n_neurons_out_actor))
        )
        recurrent_connection_out = Connection(
            source=self.layers["Y"],
            target=self.layers["Y"],
            w=w_inh_actor,
            wmin=-self.inh_out_actor,
            wmax=0,
        )
        self.add_connection(recurrent_connection_out, source="Y", target="Y")

class Diehl1LayerGroupFourierActor(Diehl1LayerFourierActor):
    def __init__(self,
                 n_actions: int = 2,
                 group_size: int = 10,
                 **kwargs):
        self.n_actions = n_actions
        self.group_size = group_size
        kwargs["n_neurons_out_actor"] = n_actions * group_size

        super().__init__(**kwargs)

    def get_action(self, spike_counts: torch.Tensor) -> int:

        sc = spike_counts.view(self.n_actions, self.group_size).sum(1)
        return torch.argmax(sc).item()


class Diehl1Layer(Network):
    # language=rst

    def __init__(
        self,
        n_inpt: int = 60,
        n_neurons_out: int = 2,
        inh_out: float = 17.5,
        dt: float = 1.0,
        nu_out: Optional[Union[float, Sequence[float]]] = (1e-4, 1e-2),
        reduction: Optional[callable] = None,
        wmin: Optional[float] = 0.0,
        wmax: Optional[float] = 1.0,
        norm: float = None,
        theta_plus: float = 0.0,
        tc_theta_decay: float = 1e7,
        inpt_shape: Optional[Iterable[int]] = None,
        w_init_out=0.5,
        thresh_out=-61.0,
        refrac_out=1,
        one_spike_out=False,
        tc_trace_input=20.0,
        tc_decay_out=15.0,
        tc_trace_out=10.0,
        tc_plus_out=10.0,
        tc_minus_out=10.0,
        tc_e_trace_out=15.0,
    ) -> None:
        # language=rst
        
        super().__init__(dt=dt)

        self.n_inpt = n_inpt
        self.inpt_shape = inpt_shape
        self.n_neurons_out = n_neurons_out
        self.inh_out = inh_out
        self.dt = dt

        self.w_init_out = w_init_out
        
        self.refrac_out = refrac_out

        self.one_spike_out = one_spike_out

        self.tc_trace_input = tc_trace_input
        self.tc_trace_out = tc_trace_out

        self.tc_decay_out = tc_decay_out

        self.tc_plus_out = tc_plus_out
        self.tc_minus_out = tc_minus_out
        self.tc_e_trace_out = tc_e_trace_out

        input_layer = Input(
            n=self.n_inpt, shape=self.inpt_shape, traces=True, tc_trace=self.tc_trace_input
        )
        self.add_layer(input_layer, name="X")

        output_layer = DiehlAndCookNodes(
            n=self.n_neurons_out,
            traces=True,
            rest=-65.0,
            reset=-65.0, # default: -60
            thresh=thresh_out, # default: -52
            refrac=self.refrac_out,
            tc_decay=self.tc_decay_out,
            tc_trace=self.tc_trace_out,
            theta_plus=0, # default: theta_plus
            tc_theta_decay=tc_theta_decay,
            one_spike=self.one_spike_out,
        )
        self.add_layer(output_layer, name="Y")

        w = self.w_init_out * torch.rand(self.n_inpt, self.n_neurons_out)
        # w = torch.normal(
        #     self.w_init_out,
        #     0.1, 
        #     size=(self.n_neurons_hid, self.n_neurons_out)
        # )
        output_connection = Connection(
            source=self.layers["X"],
            target=self.layers["Y"],
            w=w,
            update_rule=MSTDPET,
            nu=nu_out,
            reduction=reduction,
            wmin=wmin,
            wmax=wmax,
            norm=norm,
            tc_plus=self.tc_plus_out,
            tc_minus=self.tc_minus_out,
            tc_e_trace=self.tc_e_trace_out,
        )
        self.add_connection(output_connection, source="X", target="Y")

        w = -self.inh_out * (
            torch.ones(self.n_neurons_out, self.n_neurons_out)
            - torch.diag(torch.ones(self.n_neurons_out))
        )
        recurrent_connection_out = Connection(
            source=self.layers["Y"],
            target=self.layers["Y"],
            w=w,
            wmin=-self.inh_out,
            wmax=0,
        )
        self.add_connection(recurrent_connection_out, source="Y", target="Y")

class Diehl2LayersNotFC(Network):
    def __init__(
        self,
        n_inpt: int = 60,
        n_neurons_hid: int = 100,
        n_neurons_out: int = 2,
        inh_hid: float = 0.1,
        inh_out: float = 2.0,
        dt: float = 1.0,
        nu_hid: list = [0.25 * 1e-2, 1e-2], 
        nu_out: list = [0.4 * 1e-2, 1e-2], 
        reduction=None,
        wmin: float = 0.0,
        wmax: float = 1.0,
        norm: float = None,
        theta_plus: float = 0.05,
        tc_theta_decay: float = 1e7,
        inpt_shape=None,
        hidden_layer_path: str = None, 
        assignments: torch.Tensor = None,  
        w_init_hid: float = 0.5,
        w_init_out: float = 0.5,
        thresh_hid: float = -58.0,
        thresh_out: float = -61.0,
        refrac_hid: int = 5,
        refrac_out: int = 1,
    ) -> None:
        super().__init__(dt=dt)

        if assignments is None:
            raise ValueError("assignments tensor must be provided")

        self.n_inpt = n_inpt
        self.inpt_shape = inpt_shape
        self.n_neurons_hid = n_neurons_hid
        self.n_neurons_out = n_neurons_out
        self.inh_hid = inh_hid
        self.inh_out = inh_out
        self.dt = dt

        self.w_init_hid = w_init_hid
        self.w_init_out = w_init_out

        self.refrac_hid = refrac_hid
        self.refrac_out = refrac_out

        input_layer = Input(
            n=self.n_inpt, shape=self.inpt_shape, traces=True, tc_trace=20.0
        )
        self.add_layer(input_layer, name="X")

        hidden_layer = DiehlAndCookNodes(
            n=self.n_neurons_hid,
            traces=True,
            rest=-65.0,
            reset=-65.0,
            thresh=thresh_hid,
            refrac=self.refrac_hid,
            tc_decay=15.0,
            tc_trace=10.0,
            theta=0.0,
            theta_plus=theta_plus,
            tc_theta_decay=tc_theta_decay,
            one_spike=False,
        )
        self.add_layer(hidden_layer, name="H")

        if hidden_layer_path and os.path.exists(hidden_layer_path):
            loaded_net = torch.load(hidden_layer_path)
            w = loaded_net.connections['X', 'Y'].w.clone().detach()
        else:
            w = self.w_init_hid * torch.rand(self.n_inpt, self.n_neurons_hid)

        input_connection = Connection(
            source=self.layers["X"],
            target=self.layers["H"],
            w=w,
            update_rule=MSTDPET, 
            wmin=wmin,
            wmax=wmax,
            norm=norm,
        )
        self.add_connection(input_connection, source="X", target="H")

        input_connection.update_rule = NoOp(connection=input_connection)

        w_recurrent = -self.inh_hid * (
            torch.ones(self.n_neurons_hid, self.n_neurons_hid) - torch.diag(torch.ones(self.n_neurons_hid))
        )
        recurrent_connection = Connection(
            source=self.layers["H"],
            target=self.layers["H"],
            w=w_recurrent,
            wmin=-self.inh_hid,
            wmax=0.0,
        )
        self.add_connection(recurrent_connection, source="H", target="H")

        output_layer = DiehlAndCookNodes(
            n=self.n_neurons_out,
            traces=True,
            rest=-65.0,
            reset=-65.0,
            thresh=thresh_out,
            refrac=self.refrac_out,
            tc_decay=15.0,
            tc_trace=10.0,
            theta=0.0,
            theta_plus=theta_plus,
            tc_theta_decay=tc_theta_decay,
            one_spike=False,
        )
        self.add_layer(output_layer, name="Y")

        mask = torch.zeros(self.n_neurons_hid, self.n_neurons_out)
        for y_neuron in range(self.n_neurons_out):
            h_neurons = torch.nonzero(assignments == y_neuron).squeeze()
            if h_neurons.ndimension() == 0:
                h_neurons = h_neurons.unsqueeze(0)
            mask[h_neurons, y_neuron] = 1.0

        w = torch.normal(
            mean=self.w_init_out,
            std=0.01, 
            size=(self.n_neurons_hid, self.n_neurons_out)
        ) * mask
        
        output_connection = Connection(
            source=self.layers["H"],
            target=self.layers["Y"],
            w=w,
            update_rule=MSTDPET,
            nu=nu_out,
            reduction=reduction,
            wmin=wmin,
            wmax=wmax,
            norm=norm,
        )
        self.add_connection(output_connection, source="H", target="Y")

        self.mask_HY = mask

        w_recurrent_out = -self.inh_out * (
            torch.ones(self.n_neurons_out, self.n_neurons_out) - torch.diag(torch.ones(self.n_neurons_out))
        )
        recurrent_connection_out = Connection(
            source=self.layers["Y"],
            target=self.layers["Y"],
            w=w_recurrent_out,
            wmin=-self.inh_out,
            wmax=0.0,
        )
        self.add_connection(recurrent_connection_out, source="Y", target="Y")

    def apply_mask_HY(self):
        self.connections['H', 'Y'].w *= self.mask_HY


class Diehl2LayersFixedWeights(Network):
    """
    Двухслойная SNN модель с фиксированными связями между скрытым и выходным слоями на основе назначений.
    """

    def __init__(
        self,
        n_inpt: int = 60,
        n_neurons_hid: int = 100,
        n_neurons_out: int = 2,
        inh_hid: float = 0.1,
        inh_out: float = 2.0,
        dt: float = 1.0,
        wmin: float = 0.0,
        wmax: float = 1.0,
        norm: float = None,
        theta_plus: float = 0.05,
        tc_theta_decay: float = 1e7,
        inpt_shape=None,
        hidden_layer_path: str = None,  
        assignments: torch.Tensor = None, 
        w_init_hid: float = 0.5,
        thresh_hid: float = -58.0,
        thresh_out: float = -61.0,
        refrac_hid: int = 5,
        refrac_out: int = 1,
    ) -> None:
        super().__init__(dt=dt)

        if assignments is None:
            raise ValueError("assignments tensor must be provided")

        self.n_inpt = n_inpt
        self.inpt_shape = inpt_shape
        self.n_neurons_hid = n_neurons_hid
        self.n_neurons_out = n_neurons_out
        self.inh_hid = inh_hid
        self.inh_out = inh_out
        self.dt = dt

        self.w_init_hid = w_init_hid

        self.refrac_hid = refrac_hid
        self.refrac_out = refrac_out

        input_layer = Input(
            n=self.n_inpt, shape=self.inpt_shape, traces=True, tc_trace=20.0
        )
        self.add_layer(input_layer, name="X")

        hidden_layer = DiehlAndCookNodes(
            n=self.n_neurons_hid,
            traces=True,
            rest=-65.0,
            reset=-65.0,
            thresh=thresh_hid,
            refrac=self.refrac_hid,
            tc_decay=15.0,
            tc_trace=10.0,
            theta_plus=theta_plus,
            tc_theta_decay=tc_theta_decay,
            one_spike=False,
        )
        self.add_layer(hidden_layer, name="H")

        if hidden_layer_path and os.path.exists(hidden_layer_path):
            loaded_net = torch.load(hidden_layer_path)
            w = loaded_net.connections['X', 'Y'].w.clone().detach()
        else:
            w = self.w_init_hid * torch.rand(self.n_inpt, self.n_neurons_hid)

        input_connection = Connection(
            source=self.layers["X"],
            target=self.layers["H"],
            w=w,
            wmin=wmin,
            wmax=wmax,
            norm=norm,
        )
        self.add_connection(input_connection, source="X", target="H")

        input_connection.update_rule = NoOp(connection=input_connection)

        w_recurrent = -self.inh_hid * (
            torch.ones(self.n_neurons_hid, self.n_neurons_hid) - torch.diag(torch.ones(self.n_neurons_hid))
        )
        recurrent_connection = Connection(
            source=self.layers["H"],
            target=self.layers["H"],
            w=w_recurrent,
            wmin=-self.inh_hid,
            wmax=0.0,
        )
        self.add_connection(recurrent_connection, source="H", target="H")
        recurrent_connection.update_rule = NoOp(connection=recurrent_connection)

        output_layer = DiehlAndCookNodes(
            n=self.n_neurons_out,
            traces=True,
            rest=-65.0,
            reset=-65.0,
            thresh=thresh_out,
            refrac=self.refrac_out,
            tc_decay=15.0,
            tc_trace=10.0,
            theta_plus=theta_plus,
            tc_theta_decay=tc_theta_decay,
            one_spike=False,
        )
        self.add_layer(output_layer, name="Y")


        w_HY = torch.zeros(self.n_neurons_hid, self.n_neurons_out)
        for h_neuron in range(self.n_neurons_hid):
            y_neuron = assignments[h_neuron].item()
            if y_neuron == 1:
                w_HY[h_neuron, y_neuron] = 0.4
            else:
                w_HY[h_neuron, y_neuron] = 0.6

        output_connection = Connection(
            source=self.layers["H"],
            target=self.layers["Y"],
            w=w_HY,
            wmin=wmin,
            wmax=wmax,
            norm=norm,
        )
        self.add_connection(output_connection, source="H", target="Y")

        output_connection.update_rule = NoOp(connection=output_connection)

        w_recurrent_out = -self.inh_out * (
            torch.ones(self.n_neurons_out, self.n_neurons_out) - torch.diag(torch.ones(self.n_neurons_out))
        )
        recurrent_connection_out = Connection(
            source=self.layers["Y"],
            target=self.layers["Y"],
            w=w_recurrent_out,
            wmin=-self.inh_out,
            wmax=0.0,
        )
        self.add_connection(recurrent_connection_out, source="Y", target="Y")
        recurrent_connection_out.update_rule = NoOp(connection=recurrent_connection_out)

class IncreasingInhibitionNetwork(Network):
    # language=rst
    """
    Implements the inhibitory layer structure of the spiking neural network architecture
    from `(Hazan et al. 2018) <https://arxiv.org/abs/1807.09374>`_
    """

    def __init__(
        self,
        n_input: int,
        n_neurons: int = 100,
        start_inhib: float = 1.0,
        max_inhib: float = 100.0,
        dt: float = 1.0,
        nu: Optional[Union[float, Sequence[float]]] = (1e-4, 1e-2),
        reduction: Optional[callable] = None,
        wmin: float = 0.0,
        wmax: float = 1.0,
        norm: float = 78.4,
        theta_plus: float = 0.05,
        tc_theta_decay: float = 1e7,
    ) -> None:
        # language=rst
        """
        Constructor for class ``IncreasingInhibitionNetwork``.

        :param n_inpt: Number of input neurons. Matches the 1D size of the input data.
        :param n_neurons: Number of excitatory, inhibitory neurons.
        :param inh: Strength of synapse weights from inhibitory to excitatory layer.
        :param dt: Simulation time step.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events,
            respectively.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param wmin: Minimum allowed weight on input to excitatory synapses.
        :param wmax: Maximum allowed weight on input to excitatory synapses.
        :param norm: Input to excitatory layer connection weights normalization
            constant.
        :param theta_plus: On-spike increment of ``DiehlAndCookNodes`` membrane
            threshold potential.
        :param tc_theta_decay: Time constant of ``DiehlAndCookNodes`` threshold
            potential decay.
        """
        super().__init__(dt=dt)

        self.n_input = n_input
        self.n_neurons = n_neurons
        self.n_sqrt = int(np.sqrt(n_neurons))
        self.start_inhib = start_inhib
        self.max_inhib = max_inhib
        self.dt = dt

        input_layer = Input(n=self.n_input, traces=True, tc_trace=20.0)
        self.add_layer(input_layer, name="X")

        output_layer = DiehlAndCookNodes(
            n=self.n_neurons,
            traces=True,
            rest=-65.0,
            reset=-60.0,
            thresh=-52.0,
            refrac=5,
            tc_decay=100.0,
            tc_trace=20.0,
            theta_plus=theta_plus,
            tc_theta_decay=tc_theta_decay,
        )
        self.add_layer(output_layer, name="Y")

        w = 0.3 * torch.rand(self.n_input, self.n_neurons)
        input_output_conn = Connection(
            source=self.layers["X"],
            target=self.layers["Y"],
            w=w,
            update_rule=PostPre,
            nu=nu,
            reduction=reduction,
            wmin=wmin,
            wmax=wmax,
            norm=norm,
        )
        self.add_connection(input_output_conn, source="X", target="Y")

        w = torch.zeros(self.n_neurons, self.n_neurons)
        for i in range(self.n_neurons):
            for j in range(self.n_neurons):
                if i != j:
                    x1, y1 = i // self.n_sqrt, i % self.n_sqrt
                    x2, y2 = j // self.n_sqrt, j % self.n_sqrt

                    inhib = self.start_inhib * np.sqrt(euclidean([x1, y1], [x2, y2]))
                    w[i, j] = -min(self.max_inhib, inhib)

        recurrent_output_conn = Connection(
            source=self.layers["Y"],
            target=self.layers["Y"],
            w=w,
            wmin=-self.max_inhib,
            wmax=0,
        )
        self.add_connection(recurrent_output_conn, source="Y", target="Y")


class LocallyConnectedNetwork(Network):
    # language=rst
    """
    Defines a two-layer network in which the input layer is "locally connected" to the
    output layer, and the output layer is recurrently inhibited connected such that
    neurons with the same input receptive field inhibit each other.
    """

    def __init__(
        self,
        n_inpt: int,
        input_shape: List[int],
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]],
        n_filters: int,
        inh: float = 25.0,
        dt: float = 1.0,
        nu: Optional[Union[float, Sequence[float]]] = (1e-4, 1e-2),
        reduction: Optional[callable] = None,
        theta_plus: float = 0.05,
        tc_theta_decay: float = 1e7,
        wmin: float = 0.0,
        wmax: float = 1.0,
        norm: Optional[float] = 0.2,
    ) -> None:
        # language=rst
        """
        Constructor for class ``LocallyConnectedNetwork``. Uses ``DiehlAndCookNodes`` to
        avoid multiple spikes per timestep in the output layer population.

        :param n_inpt: Number of input neurons. Matches the 1D size of the input data.
        :param input_shape: Two-dimensional shape of input population.
        :param kernel_size: Size of input windows. Integer or two-tuple of integers.
        :param stride: Length of horizontal, vertical stride across input space. Integer
            or two-tuple of integers.
        :param n_filters: Number of locally connected filters per input region. Integer
            or two-tuple of integers.
        :param inh: Strength of synapse weights from output layer back onto itself.
        :param dt: Simulation time step.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events,
            respectively.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param wmin: Minimum allowed weight on ``Input`` to ``DiehlAndCookNodes``
            synapses.
        :param wmax: Maximum allowed weight on ``Input`` to ``DiehlAndCookNodes``
            synapses.
        :param theta_plus: On-spike increment of ``DiehlAndCookNodes`` membrane
            threshold potential.
        :param tc_theta_decay: Time constant of ``DiehlAndCookNodes`` threshold
            potential decay.
        :param norm: ``Input`` to ``DiehlAndCookNodes`` layer connection weights
            normalization constant.
        """
        super().__init__(dt=dt)

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)

        self.n_inpt = n_inpt
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_filters = n_filters
        self.inh = inh
        self.dt = dt
        self.theta_plus = theta_plus
        self.tc_theta_decay = tc_theta_decay
        self.wmin = wmin
        self.wmax = wmax
        self.norm = norm

        if kernel_size == input_shape:
            conv_size = [1, 1]
        else:
            conv_size = (
                int((input_shape[0] - kernel_size[0]) / stride[0]) + 1,
                int((input_shape[1] - kernel_size[1]) / stride[1]) + 1,
            )

        input_layer = Input(n=self.n_inpt, traces=True, tc_trace=20.0)

        output_layer = DiehlAndCookNodes(
            n=self.n_filters * conv_size[0] * conv_size[1],
            traces=True,
            rest=-65.0,
            reset=-60.0,
            thresh=-52.0,
            refrac=5,
            tc_decay=100.0,
            tc_trace=20.0,
            theta_plus=theta_plus,
            tc_theta_decay=tc_theta_decay,
        )
        input_output_conn = LocalConnection(
            input_layer,
            output_layer,
            kernel_size=kernel_size,
            stride=stride,
            n_filters=n_filters,
            nu=nu,
            reduction=reduction,
            update_rule=PostPre,
            wmin=wmin,
            wmax=wmax,
            norm=norm,
            input_shape=input_shape,
        )

        w = torch.zeros(n_filters, *conv_size, n_filters, *conv_size)
        for fltr1 in range(n_filters):
            for fltr2 in range(n_filters):
                if fltr1 != fltr2:
                    for i in range(conv_size[0]):
                        for j in range(conv_size[1]):
                            w[fltr1, i, j, fltr2, i, j] = -inh

        w = w.view(
            n_filters * conv_size[0] * conv_size[1],
            n_filters * conv_size[0] * conv_size[1],
        )
        recurrent_conn = Connection(output_layer, output_layer, w=w)

        self.add_layer(input_layer, name="X")
        self.add_layer(output_layer, name="Y")
        self.add_connection(input_output_conn, source="X", target="Y")
        self.add_connection(recurrent_conn, source="Y", target="Y")
