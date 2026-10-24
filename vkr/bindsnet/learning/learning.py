from abc import ABC
from typing import Union, Optional, Sequence

import torch
import numpy as np

from ..network.nodes import SRM0Nodes
from ..network.topology import (
    AbstractConnection,
    Connection,
    Conv2dConnection,
    LocalConnection,
)
from ..utils import im2col_indices


class LearningRule(ABC):
    # language=rst
    """
    Abstract base class for learning rules.
    """

    def __init__(
        self,
        connection: AbstractConnection,
        nu: Optional[Union[float, Sequence[float]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        **kwargs
    ) -> None:
        # language=rst
        """
        Abstract constructor for the ``LearningRule`` object.

        :param connection: An ``AbstractConnection`` object.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events.
        :param reduction: Method for reducing parameter updates along the batch
            dimension.
        :param weight_decay: Constant multiple to decay weights by on each iteration.
        """
        # Connection parameters.
        self.connection = connection
        self.source = connection.source
        self.target = connection.target

        self.wmin = connection.wmin
        self.wmax = connection.wmax

        # Learning rate(s).
        if nu is None:
            nu = [0.0, 0.0]
        elif isinstance(nu, float) or isinstance(nu, int):
            nu = [nu, nu]

        self.nu = nu

        # Parameter update reduction across minibatch dimension.
        if reduction is None:
            reduction = torch.mean

        self.reduction = reduction

        # Weight decay.
        self.weight_decay = weight_decay

    def update(self) -> None:
        # language=rst
        """
        Abstract method for a learning rule update.
        """
        # Implement weight decay.
        if self.weight_decay:
            self.connection.w -= self.weight_decay * self.connection.w

        # Bound weights.
        if (
            self.connection.wmin != -np.inf or self.connection.wmax != np.inf
        ) and not isinstance(self, NoOp):
            self.connection.w.clamp_(self.connection.wmin, self.connection.wmax)


class NoOp(LearningRule):
    # language=rst
    """
    Learning rule with no effect.
    """

    def __init__(
        self,
        connection: AbstractConnection,
        nu: Optional[Union[float, Sequence[float]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        **kwargs
    ) -> None:
        # language=rst
        """
        Abstract constructor for the ``LearningRule`` object.

        :param connection: An ``AbstractConnection`` object.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events.
        :param reduction: Method for reducing parameter updates along the batch
            dimension.
        :param weight_decay: Constant multiple to decay weights by on each iteration.
        """
        super().__init__(
            connection=connection,
            nu=nu,
            reduction=reduction,
            weight_decay=weight_decay,
            **kwargs
        )

    def update(self, **kwargs) -> None:
        # language=rst
        """
        Abstract method for a learning rule update.
        """
        super().update()


class PostPre(LearningRule):
    # language=rst
    """
    Simple STDP rule involving both pre- and post-synaptic spiking activity. By default,
    pre-synaptic update is negative and the post-synaptic update is positive.
    """

    def __init__(
        self,
        connection: AbstractConnection,
        nu: Optional[Union[float, Sequence[float]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        **kwargs
    ) -> None:
        # language=rst
        """
        Constructor for ``PostPre`` learning rule.

        :param connection: An ``AbstractConnection`` object whose weights the
            ``PostPre`` learning rule will modify.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events.
        :param reduction: Method for reducing parameter updates along the batch
            dimension.
        :param weight_decay: Constant multiple to decay weights by on each iteration.
        """
        super().__init__(
            connection=connection,
            nu=nu,
            reduction=reduction,
            weight_decay=weight_decay,
            **kwargs
        )

        assert (
            self.source.traces and self.target.traces
        ), "Both pre- and post-synaptic nodes must record spike traces."

        if isinstance(connection, (Connection, LocalConnection)):
            self.update = self._connection_update
        elif isinstance(connection, Conv2dConnection):
            self.update = self._conv2d_connection_update
        else:
            raise NotImplementedError(
                "This learning rule is not supported for this Connection type."
            )

    def _connection_update(self, **kwargs) -> None:
        # language=rst
        """
        Post-pre learning rule for ``Connection`` subclass of ``AbstractConnection``
        class.
        """
        batch_size = self.source.batch_size

        source_s = self.source.s.view(batch_size, -1).unsqueeze(2).float()
        source_x = self.source.x.view(batch_size, -1).unsqueeze(2)
        target_s = self.target.s.view(batch_size, -1).unsqueeze(1).float()
        target_x = self.target.x.view(batch_size, -1).unsqueeze(1)

        # Pre-synaptic update.
        if self.nu[0]:
            update = self.reduction(torch.bmm(source_s, target_x), dim=0)
            self.connection.w -= self.nu[0] * update

        # Post-synaptic update.
        if self.nu[1]:
            update = self.reduction(torch.bmm(source_x, target_s), dim=0)
            self.connection.w += self.nu[1] * update

        super().update()

    def _conv2d_connection_update(self, **kwargs) -> None:
        # language=rst
        """
        Post-pre learning rule for ``Conv2dConnection`` subclass of
        ``AbstractConnection`` class.
        """
        # Get convolutional layer parameters.
        out_channels, _, kernel_height, kernel_width = self.connection.w.size()
        padding, stride = self.connection.padding, self.connection.stride
        batch_size = self.source.batch_size

        # Reshaping spike traces and spike occurrences.
        source_x = im2col_indices(
            self.source.x, kernel_height, kernel_width, padding=padding, stride=stride
        )
        target_x = self.target.x.view(batch_size, out_channels, -1)
        source_s = im2col_indices(
            self.source.s.float(),
            kernel_height,
            kernel_width,
            padding=padding,
            stride=stride,
        )
        target_s = self.target.s.view(batch_size, out_channels, -1).float()

        # Pre-synaptic update.
        if self.nu[0]:
            pre = self.reduction(
                torch.bmm(target_x, source_s.permute((0, 2, 1))), dim=0
            )
            self.connection.w -= self.nu[0] * pre.view(self.connection.w.size())

        # Post-synaptic update.
        if self.nu[1]:
            post = self.reduction(
                torch.bmm(target_s, source_x.permute((0, 2, 1))), dim=0
            )
            self.connection.w += self.nu[1] * post.view(self.connection.w.size())

        super().update()


class WeightDependentPostPre(LearningRule):
    # language=rst
    """
    STDP rule involving both pre- and post-synaptic spiking activity. The post-synaptic
    update is positive and the pre- synaptic update is negative, and both are dependent
    on the magnitude of the synaptic weights.
    """

    def __init__(
        self,
        connection: AbstractConnection,
        nu: Optional[Union[float, Sequence[float]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        **kwargs
    ) -> None:
        # language=rst
        """
        Constructor for ``WeightDependentPostPre`` learning rule.

        :param connection: An ``AbstractConnection`` object whose weights the
            ``WeightDependentPostPre`` learning rule will modify.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events.
        :param reduction: Method for reducing parameter updates along the batch
            dimension.
        :param weight_decay: Constant multiple to decay weights by on each iteration.
        """
        super().__init__(
            connection=connection,
            nu=nu,
            reduction=reduction,
            weight_decay=weight_decay,
            **kwargs
        )

        assert self.source.traces, "Pre-synaptic nodes must record spike traces."
        assert (
            connection.wmin != -np.inf and connection.wmax != np.inf
        ), "Connection must define finite wmin and wmax."

        self.wmin = connection.wmin
        self.wmax = connection.wmax

        if isinstance(connection, (Connection, LocalConnection)):
            self.update = self._connection_update
        elif isinstance(connection, Conv2dConnection):
            self.update = self._conv2d_connection_update
        else:
            raise NotImplementedError(
                "This learning rule is not supported for this Connection type."
            )

    def _connection_update(self, **kwargs) -> None:
        # language=rst
        """
        Post-pre learning rule for ``Connection`` subclass of ``AbstractConnection``
        class.
        """
        batch_size = self.source.batch_size

        source_s = self.source.s.view(batch_size, -1).unsqueeze(2).float()
        source_x = self.source.x.view(batch_size, -1).unsqueeze(2)
        target_s = self.target.s.view(batch_size, -1).unsqueeze(1).float()
        target_x = self.target.x.view(batch_size, -1).unsqueeze(1)

        update = 0

        # Pre-synaptic update.
        if self.nu[0]:
            outer_product = self.reduction(torch.bmm(source_s, target_x), dim=0)
            update -= self.nu[0] * outer_product * (self.connection.w - self.wmin)

        # Post-synaptic update.
        if self.nu[1]:
            outer_product = self.reduction(torch.bmm(source_x, target_s), dim=0)
            update += self.nu[1] * outer_product * (self.wmax - self.connection.w)

        self.connection.w += update

        super().update()

    def _conv2d_connection_update(self, **kwargs) -> None:
        # language=rst
        """
        Post-pre learning rule for ``Conv2dConnection`` subclass of
        ``AbstractConnection`` class.
        """
        # Get convolutional layer parameters.
        (
            out_channels,
            in_channels,
            kernel_height,
            kernel_width,
        ) = self.connection.w.size()
        padding, stride = self.connection.padding, self.connection.stride
        batch_size = self.source.batch_size

        # Reshaping spike traces and spike occurrences.
        source_x = im2col_indices(
            self.source.x, kernel_height, kernel_width, padding=padding, stride=stride
        )
        target_x = self.target.x.view(batch_size, out_channels, -1)
        source_s = im2col_indices(
            self.source.s.float(),
            kernel_height,
            kernel_width,
            padding=padding,
            stride=stride,
        )
        target_s = self.target.s.view(batch_size, out_channels, -1).float()

        update = 0

        # Pre-synaptic update.
        if self.nu[0]:
            pre = self.reduction(
                torch.bmm(target_x, source_s.permute((0, 2, 1))), dim=0
            )
            update -= (
                self.nu[0]
                * pre.view(self.connection.w.size())
                * (self.connection.w - self.wmin)
            )

        # Post-synaptic update.
        if self.nu[1]:
            post = self.reduction(
                torch.bmm(target_s, source_x.permute((0, 2, 1))), dim=0
            )
            update += (
                self.nu[1]
                * post.view(self.connection.w.size())
                * (self.wmax - self.connection.wmin)
            )

        self.connection.w += update

        super().update()


class Hebbian(LearningRule):
    # language=rst
    """
    Simple Hebbian learning rule. Pre- and post-synaptic updates are both positive.
    """

    def __init__(
        self,
        connection: AbstractConnection,
        nu: Optional[Union[float, Sequence[float]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        **kwargs
    ) -> None:
        # language=rst
        """
        Constructor for ``Hebbian`` learning rule.

        :param connection: An ``AbstractConnection`` object whose weights the
            ``Hebbian`` learning rule will modify.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events.
        :param reduction: Method for reducing parameter updates along the batch
            dimension.
        :param weight_decay: Constant multiple to decay weights by on each iteration.
        """
        super().__init__(
            connection=connection,
            nu=nu,
            reduction=reduction,
            weight_decay=weight_decay,
            **kwargs
        )

        assert (
            self.source.traces and self.target.traces
        ), "Both pre- and post-synaptic nodes must record spike traces."

        if isinstance(connection, (Connection, LocalConnection)):
            self.update = self._connection_update
        elif isinstance(connection, Conv2dConnection):
            self.update = self._conv2d_connection_update
        else:
            raise NotImplementedError(
                "This learning rule is not supported for this Connection type."
            )

    def _connection_update(self, **kwargs) -> None:
        # language=rst
        """
        Hebbian learning rule for ``Connection`` subclass of ``AbstractConnection``
        class.
        """
        batch_size = self.source.batch_size

        source_s = self.source.s.view(batch_size, -1).unsqueeze(2).float()
        source_x = self.source.x.view(batch_size, -1).unsqueeze(2)
        target_s = self.target.s.view(batch_size, -1).unsqueeze(1).float()
        target_x = self.target.x.view(batch_size, -1).unsqueeze(1)

        # Pre-synaptic update.
        update = self.reduction(torch.bmm(source_s, target_x), dim=0)
        self.connection.w += self.nu[0] * update

        # Post-synaptic update.
        update = self.reduction(torch.bmm(source_x, target_s), dim=0)
        self.connection.w += self.nu[1] * update

        super().update()

    def _conv2d_connection_update(self, **kwargs) -> None:
        # language=rst
        """
        Hebbian learning rule for ``Conv2dConnection`` subclass of
        ``AbstractConnection`` class.
        """
        out_channels, _, kernel_height, kernel_width = self.connection.w.size()
        padding, stride = self.connection.padding, self.connection.stride
        batch_size = self.source.batch_size

        # Reshaping spike traces and spike occurrences.
        source_x = im2col_indices(
            self.source.x, kernel_height, kernel_width, padding=padding, stride=stride
        )
        target_x = self.target.x.view(batch_size, out_channels, -1)
        source_s = im2col_indices(
            self.source.s.float(),
            kernel_height,
            kernel_width,
            padding=padding,
            stride=stride,
        )
        target_s = self.target.s.view(batch_size, out_channels, -1).float()

        # Pre-synaptic update.
        pre = self.reduction(torch.bmm(target_x, source_s.permute((0, 2, 1))), dim=0)
        self.connection.w += self.nu[0] * pre.view(self.connection.w.size())

        # Post-synaptic update.
        post = self.reduction(torch.bmm(target_s, source_x.permute((0, 2, 1))), dim=0)
        self.connection.w += self.nu[1] * post.view(self.connection.w.size())

        super().update()


class MSTDP(LearningRule):
    # language=rst
    """
    Reward-modulated STDP. Adapted from `(Florian 2007)
    <https://florian.io/papers/2007_Florian_Modulated_STDP.pdf>`_.
    """

    def __init__(
        self,
        connection: AbstractConnection,
        nu: Optional[Union[float, Sequence[float]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        **kwargs
    ) -> None:
        # language=rst
        """
        Constructor for ``MSTDP`` learning rule.

        :param connection: An ``AbstractConnection`` object whose weights the ``MSTDP``
            learning rule will modify.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events,
            respectively.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param weight_decay: Constant multiple to decay weights by on each iteration.

        Keyword arguments:

        :param tc_plus: Time constant for pre-synaptic firing trace.
        :param tc_minus: Time constant for post-synaptic firing trace.
        """
        super().__init__(
            connection=connection,
            nu=nu,
            reduction=reduction,
            weight_decay=weight_decay,
            **kwargs
        )

        if isinstance(connection, (Connection, LocalConnection)):
            self.update = self._connection_update
        elif isinstance(connection, Conv2dConnection):
            self.update = self._conv2d_connection_update
        else:
            raise NotImplementedError(
                "This learning rule is not supported for this Connection type."
            )

        self.tc_plus = torch.tensor(kwargs.get("tc_plus", 20.0))
        self.tc_minus = torch.tensor(kwargs.get("tc_minus", 20.0))

    def _connection_update(self, **kwargs) -> None:
        # language=rst
        """
        MSTDP learning rule for ``Connection`` subclass of ``AbstractConnection`` class.

        Keyword arguments:

        :param Union[float, torch.Tensor] reward: Reward signal from reinforcement
            learning task.
        :param float a_plus: Learning rate (post-synaptic).
        :param float a_minus: Learning rate (pre-synaptic).
        """
        batch_size = self.source.batch_size

        # Initialize eligibility, P^+, and P^-.
        if not hasattr(self, "p_plus"):
            self.p_plus = torch.zeros(batch_size, *self.source.shape)
        if not hasattr(self, "p_minus"):
            self.p_minus = torch.zeros(batch_size, *self.target.shape)
        if not hasattr(self, "eligibility"):
            self.eligibility = torch.zeros(batch_size, *self.connection.w.shape)

        # Reshape pre- and post-synaptic spikes.
        source_s = self.source.s.view(batch_size, -1).float()
        target_s = self.target.s.view(batch_size, -1).float()

        # Parse keyword arguments.
        reward = kwargs["reward"]
        a_plus = torch.tensor(kwargs.get("a_plus", 1.0))
        a_minus = torch.tensor(kwargs.get("a_minus", -1.0))

        # Compute weight update based on the eligibility value of the past timestep.
        update = reward * self.eligibility
        self.connection.w += self.nu[0] * self.reduction(update, dim=0)

        # Update P^+ and P^- values.
        self.p_plus *= torch.exp(-self.connection.dt / self.tc_plus)
        self.p_plus += a_plus * source_s
        self.p_minus *= torch.exp(-self.connection.dt / self.tc_minus)
        self.p_minus += a_minus * target_s

        # Calculate point eligibility value.
        self.eligibility = torch.bmm(
            self.p_plus.unsqueeze(2), target_s.unsqueeze(1)
        ) + torch.bmm(source_s.unsqueeze(2), self.p_minus.unsqueeze(1))

        super().update()

    def _conv2d_connection_update(self, **kwargs) -> None:
        # language=rst
        """
        MSTDP learning rule for ``Conv2dConnection`` subclass of ``AbstractConnection``
        class.

        Keyword arguments:

        :param Union[float, torch.Tensor] reward: Reward signal from reinforcement
            learning task.
        :param float a_plus: Learning rate (post-synaptic).
        :param float a_minus: Learning rate (pre-synaptic).
        """
        batch_size = self.source.batch_size

        # Initialize eligibility.
        if not hasattr(self, "eligibility"):
            self.eligibility = torch.zeros(batch_size, *self.connection.w.shape)

        # Parse keyword arguments.
        reward = kwargs["reward"]
        a_plus = torch.tensor(kwargs.get("a_plus", 1.0))
        a_minus = torch.tensor(kwargs.get("a_minus", -1.0))

        batch_size = self.source.batch_size

        # Compute weight update based on the eligibility value of the past timestep.
        update = reward * self.eligibility
        self.connection.w += self.nu[0] * torch.sum(update, dim=0)

        out_channels, _, kernel_height, kernel_width = self.connection.w.size()
        padding, stride = self.connection.padding, self.connection.stride

        # Initialize P^+ and P^-.
        if not hasattr(self, "p_plus"):
            self.p_plus = torch.zeros(batch_size, *self.source.shape)
            self.p_plus = im2col_indices(
                self.p_plus, kernel_height, kernel_width, padding=padding, stride=stride
            )
        if not hasattr(self, "p_minus"):
            self.p_minus = torch.zeros(batch_size, *self.target.shape)
            self.p_minus = self.p_minus.view(batch_size, out_channels, -1).float()

        # Reshaping spike occurrences.
        source_s = im2col_indices(
            self.source.s.float(),
            kernel_height,
            kernel_width,
            padding=padding,
            stride=stride,
        )
        target_s = self.target.s.view(batch_size, out_channels, -1).float()

        # Update P^+ and P^- values.
        self.p_plus *= torch.exp(-self.connection.dt / self.tc_plus)
        self.p_plus += a_plus * source_s
        self.p_minus *= torch.exp(-self.connection.dt / self.tc_minus)
        self.p_minus += a_minus * target_s

        # Calculate point eligibility value.
        self.eligibility = torch.bmm(
            target_s, self.p_plus.permute((0, 2, 1))
        ) + torch.bmm(self.p_minus, source_s.permute((0, 2, 1)))
        self.eligibility = self.eligibility.view(self.connection.w.size())

        super().update()

class MSTDPET(LearningRule):
    # language=rst
    """
    Reward-modulated STDP with eligibility trace. Adapted from
    `(Florian 2007) <https://florian.io/papers/2007_Florian_Modulated_STDP.pdf>`_.
    """

    def __init__(
        self,
        connection: AbstractConnection,
        nu: Optional[Union[float, Sequence[float]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        **kwargs
    ) -> None:
        # language=rst
        """
        Constructor for ``MSTDPET`` learning rule.

        :param connection: An ``AbstractConnection`` object whose weights the
            ``MSTDPET`` learning rule will modify.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events,
            respectively.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param weight_decay: Constant multiple to decay weights by on each iteration.

        Keyword arguments:

        :param float tc_plus: Time constant for pre-synaptic firing trace.
        :param float tc_minus: Time constant for post-synaptic firing trace.
        :param float tc_e_trace: Time constant for the eligibility trace.
        """
        super().__init__(
            connection=connection,
            nu=nu,
            reduction=reduction,
            weight_decay=weight_decay,
            **kwargs
        )

        if isinstance(connection, (Connection, LocalConnection)):
            self.update = self._connection_update
        elif isinstance(connection, Conv2dConnection):
            self.update = self._conv2d_connection_update
        else:
            raise NotImplementedError(
                "This learning rule is not supported for this Connection type."
            )

        self.tc_plus = torch.tensor(kwargs.get("tc_plus", 20.0))
        self.tc_minus = torch.tensor(kwargs.get("tc_minus", 20.0))
        self.tc_e_trace = torch.tensor(kwargs.get("tc_e_trace", 25.0))

    def _connection_update(self, step=0, **kwargs) -> None:
        # language=rst
        """
        MSTDPET learning rule for ``Connection`` subclass of ``AbstractConnection``
        class.

        Keyword arguments:

        :param Union[float, torch.Tensor] reward: Reward signal from reinforcement
            learning task.
        :param float a_plus: Learning rate (post-synaptic).
        :param float a_minus: Learning rate (pre-synaptic).
        """
        # Initialize eligibility, eligibility trace, P^+, and P^-.
        if not hasattr(self, "p_plus"):
            self.p_plus = torch.zeros(self.source.n)
        if not hasattr(self, "p_minus"):
            self.p_minus = torch.zeros(self.target.n)
        if not hasattr(self, "eligibility"):
            self.eligibility = torch.zeros(*self.connection.w.shape)
        if not hasattr(self, "eligibility_trace"):
            self.eligibility_trace = torch.zeros(*self.connection.w.shape)

        # Reshape pre- and post-synaptic spikes.
        source_s = (self.source.s).reshape(-1).float()
        target_s = (self.target.s).reshape(-1).float()

        # Parse keyword arguments.
        reward = kwargs["reward"]
        if isinstance(reward, dict):
            reward = reward[self.connection]
        # a_plus = torch.tensor(kwargs.get("a_plus", 1.0))
        # a_minus = torch.tensor(kwargs.get("a_minus", -1.0))

        # --- a_plus ---
        a_plus = kwargs.get("a_plus", getattr(self, "a_plus", 1.0))
        if isinstance(a_plus, dict):
            a_plus = a_plus.get(self.connection, getattr(self, "a_plus", 1.0))
        a_plus = torch.tensor(a_plus)

        # --- a_minus ---
        a_minus = kwargs.get("a_minus", getattr(self, "a_minus", -1.0))
        if isinstance(a_minus, dict):
            a_minus = a_minus.get(self.connection, getattr(self, "a_minus", -1.0))
        a_minus = torch.tensor(a_minus)


        # Calculate value of eligibility trace based on the value
        # of the point eligibility value of the past timestep.
        self.eligibility_trace *= torch.exp(-self.connection.dt / self.tc_e_trace)
        self.eligibility_trace += self.eligibility / self.tc_e_trace

        # Compute weight update.
        # changed self.nu[0] to self.nu[1]
        update = self.nu[1] * self.connection.dt * reward * self.eligibility_trace
        self.connection.w += (
            self.nu[1] * self.connection.dt * reward * self.eligibility_trace
        )

        # Update P^+ and P^- values.
        self.p_plus *= torch.exp(-self.connection.dt / self.tc_plus)
        self.p_plus += a_plus * source_s
        self.p_minus *= torch.exp(-self.connection.dt / self.tc_minus)
        self.p_minus += a_minus * target_s

        # Calculate point eligibility value.
        self.eligibility = torch.ger(self.p_plus, target_s) + torch.ger(
            source_s, self.p_minus
        )

        super().update()

    def _conv2d_connection_update(self, **kwargs) -> None:
        # language=rst
        """
        MSTDPET learning rule for ``Conv2dConnection`` subclass of
        ``AbstractConnection`` class.

        Keyword arguments:

        :param Union[float, torch.Tensor] reward: Reward signal from reinforcement
            learning task.
        :param float a_plus: Learning rate (post-synaptic).
        :param float a_minus: Learning rate (pre-synaptic).
        """
        batch_size = self.source.batch_size

        # Initialize eligibility and eligibility trace.
        if not hasattr(self, "eligibility"):
            self.eligibility = torch.zeros(batch_size, *self.connection.w.shape)
        if not hasattr(self, "eligibility_trace"):
            self.eligibility_trace = torch.zeros(batch_size, *self.connection.w.shape)

        # Parse keyword arguments.
        reward = kwargs["reward"]
        a_plus = torch.tensor(kwargs.get("a_plus", 1.0))
        a_minus = torch.tensor(kwargs.get("a_minus", -1.0))

        # Calculate value of eligibility trace based on the value
        # of the point eligibility value of the past timestep.
        self.eligibility_trace *= torch.exp(-self.connection.dt / self.tc_e_trace)

        # Compute weight update.
        update = reward * self.eligibility_trace
        self.connection.w += self.nu[0] * self.connection.dt * torch.sum(update, dim=0)

        out_channels, _, kernel_height, kernel_width = self.connection.w.size()
        padding, stride = self.connection.padding, self.connection.stride

        # Initialize P^+ and P^-.
        if not hasattr(self, "p_plus"):
            self.p_plus = torch.zeros(batch_size, *self.source.shape)
            self.p_plus = im2col_indices(
                self.p_plus, kernel_height, kernel_width, padding=padding, stride=stride
            )
        if not hasattr(self, "p_minus"):
            self.p_minus = torch.zeros(batch_size, *self.target.shape)
            self.p_minus = self.p_minus.view(batch_size, out_channels, -1).float()

        # Reshaping spike occurrences.
        source_s = im2col_indices(
            self.source.s.float(),
            kernel_height,
            kernel_width,
            padding=padding,
            stride=stride,
        )
        target_s = (
            self.target.s.permute(1, 2, 3, 0).view(batch_size, out_channels, -1).float()
        )

        # Update P^+ and P^- values.
        self.p_plus *= torch.exp(-self.connection.dt / self.tc_plus)
        self.p_plus += a_plus * source_s
        self.p_minus *= torch.exp(-self.connection.dt / self.tc_minus)
        self.p_minus += a_minus * target_s

        # Calculate point eligibility value.
        self.eligibility = torch.bmm(
            target_s, self.p_plus.permute((0, 2, 1))
        ) + torch.bmm(self.p_minus, source_s.permute((0, 2, 1)))
        self.eligibility = self.eligibility.view(self.connection.w.size())

        super().update()

    def reset_state_variables(self):
        if hasattr(self, "p_plus"):  self.p_plus = torch.zeros(self.source.n)
        if hasattr(self, "p_minus"):  self.p_minus = torch.zeros(self.target.n)
        if hasattr(self, "eligibility"): self.eligibility = torch.zeros(*self.connection.w.shape)
        if hasattr(self, "eligibility_trace"):  self.eligibility_trace = torch.zeros(*self.connection.w.shape) # here was the bug
        # there was no assigning to self.eligibility_trace



class MSTDPETPrevUpd(MSTDPET):
    def _connection_update(self, step=0, **kwargs) -> None:
        # language=rst
        """
        MSTDPET learning rule for ``Connection`` subclass of ``AbstractConnection``
        class.

        Keyword arguments:

        :param Union[float, torch.Tensor] reward: Reward signal from reinforcement
            learning task.
        :param float a_plus: Learning rate (post-synaptic).
        :param float a_minus: Learning rate (pre-synaptic).
        """
        # Initialize eligibility, eligibility trace, P^+, and P^-.
        if not hasattr(self, "p_plus"):
            self.p_plus = torch.zeros(self.source.n)
        if not hasattr(self, "p_minus"):
            self.p_minus = torch.zeros(self.target.n)
        if not hasattr(self, "eligibility"):
            self.eligibility = torch.zeros(*self.connection.w.shape)
        if not hasattr(self, "eligibility_trace"):
            self.eligibility_trace = torch.zeros(*self.connection.w.shape)

        # Reshape pre- and post-synaptic spikes.
        source_s = (self.source.s).reshape(-1).float()
        target_s = (self.target.s).reshape(-1).float()

        # Parse keyword arguments.
        reward = kwargs["reward"]
        if isinstance(reward, dict):
            reward = reward[self.connection]
        a_plus = torch.tensor(kwargs.get("a_plus", 1.0))
        a_minus = torch.tensor(kwargs.get("a_minus", -1.0))

        # Compute weight update.
        # changed self.nu[0] to self.nu[1]
        update = self.nu[1] * self.connection.dt * reward * self.eligibility_trace
        self.connection.w += (
            self.nu[1] * self.connection.dt * reward * self.eligibility_trace
        )

        # Calculate value of eligibility trace based on the value
        # of the point eligibility value of the past timestep.
        self.eligibility_trace *= torch.exp(-self.connection.dt / self.tc_e_trace)
        self.eligibility_trace += self.eligibility / self.tc_e_trace


        # Update P^+ and P^- values.
        self.p_plus *= torch.exp(-self.connection.dt / self.tc_plus)
        self.p_plus += a_plus * source_s
        self.p_minus *= torch.exp(-self.connection.dt / self.tc_minus)
        self.p_minus += a_minus * target_s

        # Calculate point eligibility value.
        self.eligibility = torch.ger(self.p_plus, target_s) + torch.ger(
            source_s, self.p_minus
        )

        super().update()
 

class SourceFBTDSTDP(LearningRule):
    # language=rst
    """
    Feedback-modulated TD-STDP.
    """

    def __init__(
        self,
        connection: AbstractConnection,
        nu: Optional[Union[float, Sequence[float]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        **kwargs
    ) -> None:
        # language=rst
        """
        Constructor for ``FBTDSTDP`` learning rule.

        :param connection: An ``AbstractConnection`` object whose weights the ``MSTDP``
            learning rule will modify.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events,
            respectively.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param weight_decay: Constant multiple to decay weights by on each iteration.

        Keyword arguments:

        :param tc_plus: Time constant for pre-synaptic firing trace.
        :param tc_minus: Time constant for post-synaptic firing trace.
        :param tc_e_trace: Time constant for eligibility trace
        :param tc_a_trace: Time constant for feedback-gated eligibility trace
        :param fb_gate: Boolean whether to gate by feedback signal. If not then just TD-STDP
        :param targ_firing_r: If larger than 0, then it will add target firing rate subtracting
            current firing rate as reward, scaled by targ_firing_r, for all post-synaptic unit
        :param targ_firing_rate: Target firing rate of post-synpatic units
        :param inh: Whether the target neuron is inhibitory; only affect entropy grad sign in updating
        :param adam: Whether to use adam optimizer
        :param adam_beta_1: Beta1 in adam optimizer
        :param adam_beta_2: Beta2 in adam optimizer
        :param gpu: Whether to use gpu
        """
        super().__init__(
            connection=connection,
            nu=nu,
            reduction=reduction,
            weight_decay=weight_decay,
            **kwargs
        )

        if isinstance(connection, (Connection, LocalConnection)):
            self.update = self._connection_update
        else:
            raise NotImplementedError(
                "This learning rule is not supported for this Connection type."
            )
        self.tc_plus = torch.tensor(kwargs.get("tc_plus", 20.0))
        self.tc_minus = torch.tensor(kwargs.get("tc_minus", 20.0))
        self.tc_e_trace = torch.tensor(kwargs.get("tc_e_trace", 20.0))
        self.tc_a_trace = torch.tensor(kwargs.get("tc_a_trace", 20.0))
        self.fb_gate = kwargs.get("fb_gate", False)

        if self.fb_gate:
            self.num_actions = kwargs["num_actions"]
            assert self.target.n % self.num_actions == 0, "Number of node has to be divisible by number of actions"
            self.node_label = torch.repeat_interleave(torch.arange(start=0, end=self.num_actions), 
                self.target.n//self.num_actions).unsqueeze(0)     
            if kwargs["gpu"]: self.node_label = self.node_label.cuda()

        self.targ_firing_r = torch.tensor(kwargs.get("targ_firing_r", 0))        
        self.targ_firing_rate = torch.tensor(kwargs.get("targ_firing_rate", 0.2))        
        self.inh = kwargs.get("inh", False)
        self.adam = kwargs.get("adam", False)
        adam_beta_1 = kwargs.get("adam_beta_1", 0.999)
        adam_beta_2 = kwargs.get("adam_beta_2", 0.99999)
        self.weight_reg = kwargs.get("weight_reg", 0)
        self.gpu = kwargs.get("gpu", False)

        if self.adam: self.optimzier = adam_optimizer(nu[1], beta_1=adam_beta_1, beta_2=adam_beta_2)

    def _connection_update(self, **kwargs) -> None:
        # language=rst
        """
        FB-modulated TD-STDP learning rule for ``Connection`` subclass of ``AbstractConnection`` class.

        Keyword arguments:

        :param Union[float, torch.Tensor] reward: Reward signal from reinforcement
            learning task.
        :param float a_plus: Learning rate (post-synaptic).
        :param float a_minus: Learning rate (pre-synaptic).
        """
        batch_size = self.source.batch_size
        gpu = kwargs.get("gpu", False)

        self.at_decay = torch.exp(-self.connection.dt / self.tc_a_trace)
        self.et_decay = torch.exp(-self.connection.dt / self.tc_e_trace)
        self.ltp_decay = torch.exp(-self.connection.dt / self.tc_plus)
        self.ltd_decay = torch.exp(-self.connection.dt / self.tc_minus)

        # Initialize eligibility, P^+, and P^-.
        if not hasattr(self, "p_plus"):
            self.p_plus = torch.zeros(batch_size, int(np.prod(self.source.shape)))
            if gpu: self.p_plus = self.p_plus.cuda()
        if not hasattr(self, "p_minus"):
            self.p_minus = torch.zeros(batch_size, *self.target.shape)
            if gpu: self.p_minus = self.p_minus.cuda()
        if not hasattr(self, "eligibility"):
            self.eligibility = torch.zeros(batch_size, *self.connection.w.shape)
            if gpu: self.eligibility = self.eligibility.cuda()
        if not hasattr(self, "eligibility_trace"):
            self.eligibility_trace = torch.zeros(batch_size, *self.connection.w.shape)
            if gpu: self.eligibility_trace = self.eligibility_trace.cuda() 
        if self.fb_gate:     
            if not hasattr(self, "action_v"):
                self.action_v = torch.zeros(batch_size, *self.connection.w.shape)
                if gpu: self.action_v = self.action_v.cuda()            
            if not hasattr(self, "action_trace"):
                self.action_trace = torch.zeros(batch_size, *self.connection.w.shape)
                if gpu: self.action_trace = self.action_trace.cuda()

        # Reshape pre- and post-synaptic spikes.
        source_s = self.source.s.view(batch_size, -1).float()
        target_s = self.target.s.view(batch_size, -1).float()        
        reward = kwargs["reward"][self.connection]

        if self.fb_gate:   
            p = torch.repeat_interleave(kwargs["p"], self.target.n//self.num_actions, dim=-1).unsqueeze(1)            
            a = (kwargs["action"].unsqueeze(-1) == self.node_label).unsqueeze(1).float()     
            self.action_v = (a - p) * self.eligibility_trace
            self.action_trace *= self.at_decay
            self.action_trace += self.action_v / self.tc_a_trace
            update = reward * self.action_trace

            if kwargs.get("ent", None) is not None:
                ent = torch.repeat_interleave(kwargs["ent"], self.target.n//self.num_actions, dim=-1).unsqueeze(1)
                if self.inh: 
                    update -= ent*self.eligibility_trace
                else:
                    update += ent*self.eligibility_trace            
        else:
            update = reward * self.eligibility_trace

        if self.targ_firing_r != 0:
            targ_adj = (self.targ_firing_rate - torch.mean(self.target.x, dim=1)/self.target.tc_trace)*self.targ_firing_r
            update += targ_adj.unsqueeze(-1).unsqueeze(-1)*self.eligibility_trace
        
        if self.weight_reg != 0:
            update -= 0.5 * self.weight_reg * self.connection.w

        if self.adam: 
            update = self.optimzier.delta([self.reduction(update, dim=0)])[0]
        else:
            update = self.reduction(update, dim=0)
            
        self.connection.w += update

        # Update P^+ and P^- values.
        self.p_plus *= self.ltp_decay
        self.p_plus += self.nu[1] * source_s
        # Calculate point eligibility value.
        self.eligibility = torch.bmm(self.p_plus.unsqueeze(2), target_s.unsqueeze(1))

        if self.nu[0] != 0:
            self.p_minus *= self.ltd_decay
            self.p_minus += -self.nu[0] * target_s
            self.eligibility += torch.bmm(source_s.unsqueeze(2), self.p_minus.unsqueeze(1))

        # Compute weight update based on the eligibility value of the past timestep.          
        self.eligibility_trace *= self.et_decay
        self.eligibility_trace += self.eligibility / self.tc_e_trace
        
        super().update()

    def reset_state_variables(self, index=slice(None)) -> None:
        #super().reset_state_variables()      
        if hasattr(self, "p_plus"):  self.p_plus[index] = 0
        if hasattr(self, "p_minus"):  self.p_minus[index] = 0
        if hasattr(self, "eligibility_trace"):  self.eligibility_trace[index] = 0
        if self.fb_gate and hasattr(self, "action_trace"): self.action_trace[index] = 0

    def set_batch_size(self, batch_size) -> None:
        if hasattr(self, "p_plus"): delattr(self, "p_plus")
        if hasattr(self, "p_minus"): delattr(self, "p_minus")
        if hasattr(self, "eligibility"): delattr(self, "eligibility")
        if hasattr(self, "eligibility_trace"): delattr(self, "eligibility_trace")
        if hasattr(self, "action_v"): delattr(self, "action_v")
        if hasattr(self, "action_trace"): delattr(self, "action_trace")



class FBTDSTDP(LearningRule):
    # language=rst
    """
    Feedback-modulated TD-STDP.
    """

    def __init__(
        self,
        connection,
        nu=None,
        reduction=None,
        weight_decay: float = 0.0,
        **kwargs
    ) -> None:
        # language=rst
        """
        Constructor for ``FBTDSTDP`` learning rule.

        :param connection: An ``AbstractConnection`` object whose weights the ``MSTDP``
            learning rule will modify.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events,
            respectively.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param weight_decay: Constant multiple to decay weights by on each iteration.

        Keyword arguments:

        :param tc_plus: Time constant for pre-synaptic firing trace.
        :param tc_minus: Time constant for post-synaptic firing trace.
        :param tc_e_trace: Time constant for eligibility trace
        :param tc_a_trace: Time constant for feedback-gated eligibility trace
        :param fb_gate: Boolean whether to gate by feedback signal. If not then just TD-STDP
        :param targ_firing_r: If larger than 0, then it will add target firing rate subtracting
            current firing rate as reward, scaled by targ_firing_r, for all post-synaptic unit
        :param targ_firing_rate: Target firing rate of post-synpatic units
        :param inh: Whether the target neuron is inhibitory; only affect entropy grad sign in updating
        :param adam: Whether to use adam optimizer
        :param adam_beta_1: Beta1 in adam optimizer
        :param adam_beta_2: Beta2 in adam optimizer
        :param gpu: Whether to use gpu
        """
        super().__init__(
            connection=connection,
            nu=nu,
            reduction=reduction,
            weight_decay=weight_decay,
            **kwargs
        )

        if isinstance(connection, (Connection, LocalConnection)):
            self.update = self._connection_update
        else:
            raise NotImplementedError(
                "This learning rule is not supported for this Connection type."
            )
        self.tc_plus = torch.tensor(kwargs.get("tc_plus", 20.0))
        self.tc_minus = torch.tensor(kwargs.get("tc_minus", 20.0))
        self.tc_e_trace = torch.tensor(kwargs.get("tc_e_trace", 20.0))
        self.tc_a_trace = torch.tensor(kwargs.get("tc_a_trace", 20.0))
        self.fb_gate = kwargs.get("fb_gate", False)

        if self.fb_gate:
            self.num_actions = kwargs["num_actions"]
            assert self.target.n % self.num_actions == 0, "Number of node has to be divisible by number of actions"
            self.node_label = torch.repeat_interleave(torch.arange(start=0, end=self.num_actions), 
                self.target.n//self.num_actions).unsqueeze(0)     



    def _connection_update(self, **kwargs) -> None:
        # language=rst
        """
        FB-modulated TD-STDP learning rule for ``Connection`` subclass of ``AbstractConnection`` class.

        Keyword arguments:

        :param Union[float, torch.Tensor] reward: Reward signal from reinforcement
            learning task.
        :param float a_plus: Learning rate (post-synaptic).
        :param float a_minus: Learning rate (pre-synaptic).
        """
        batch_size = self.source.batch_size

        self.at_decay = torch.exp(-self.connection.dt / self.tc_a_trace)
        self.et_decay = torch.exp(-self.connection.dt / self.tc_e_trace)
        self.ltp_decay = torch.exp(-self.connection.dt / self.tc_plus)
        self.ltd_decay = torch.exp(-self.connection.dt / self.tc_minus)

        # Initialize eligibility, P^+, and P^-.
        if not hasattr(self, "p_plus"):
            self.p_plus = torch.zeros(int(np.prod(self.source.shape)))
        if not hasattr(self, "p_minus"):
            self.p_minus = torch.zeros(*self.target.shape)
        if not hasattr(self, "eligibility"):
            self.eligibility = torch.zeros(*self.connection.w.shape)
        if not hasattr(self, "eligibility_trace"):
            self.eligibility_trace = torch.zeros(*self.connection.w.shape)
        if self.fb_gate:     
            if not hasattr(self, "action_v"):
                self.action_v = torch.zeros(*self.connection.w.shape)
            if not hasattr(self, "action_trace"):
                self.action_trace = torch.zeros(*self.connection.w.shape)

        # Reshape pre- and post-synaptic spikes.
        source_s = self.source.s.view(-1).float()
        target_s = self.target.s.view(-1).float()        
        # reward = kwargs["reward"][self.connection]
        reward = kwargs["reward"]

        if self.fb_gate:   
            p = torch.repeat_interleave(kwargs["p"], self.target.n//self.num_actions, dim=-1)           
            a = (kwargs["action"].unsqueeze(-1) == self.node_label).float()     
            self.action_v = (a - p) * self.eligibility_trace
            self.action_trace *= self.at_decay
            self.action_trace += self.action_v / self.tc_a_trace
            update = reward * self.action_trace
         
        else:
            update = reward * self.eligibility_trace
            
        self.connection.w += update

        # Update P^+ and P^- values.
        self.p_plus *= self.ltp_decay
        self.p_plus += self.nu[1] * source_s
        # Calculate point eligibility value.
        # self.eligibility = torch.bmm(self.p_plus.unsqueeze(2), target_s.unsqueeze(1))
        self.eligibility = torch.ger(self.p_plus, target_s)

        if self.nu[0] != 0:
            self.p_minus *= self.ltd_decay
            self.p_minus += -self.nu[0] * target_s
            # self.eligibility += torch.bmm(source_s.unsqueeze(2), self.p_minus.unsqueeze(1))
            self.eligibility += torch.ger(source_s, self.p_minus)

        # Compute weight update based on the eligibility value of the past timestep.          
        self.eligibility_trace *= self.et_decay
        self.eligibility_trace += self.eligibility / self.tc_e_trace
        
        super().update()

    def reset_state_variables(self, index=slice(None)) -> None:
        #super().reset_state_variables()      
        if hasattr(self, "p_plus"):  self.p_plus[index] = 0
        if hasattr(self, "p_minus"):  self.p_minus[index] = 0
        if hasattr(self, "eligibility_trace"):  self.eligibility_trace[index] = 0
        if self.fb_gate and hasattr(self, "action_trace"): self.action_trace[index] = 0

    def set_batch_size(self, batch_size) -> None:
        if hasattr(self, "p_plus"): delattr(self, "p_plus")
        if hasattr(self, "p_minus"): delattr(self, "p_minus")
        if hasattr(self, "eligibility"): delattr(self, "eligibility")
        if hasattr(self, "eligibility_trace"): delattr(self, "eligibility_trace")
        if hasattr(self, "action_v"): delattr(self, "action_v")
        if hasattr(self, "action_trace"): delattr(self, "action_trace")


class MSTDPETCustom(LearningRule):
    """
    Custom Reward-modulated STDP with eligibility trace.
    
    This learning rule is adapted from MSTDPET but allows you to pass in a reward
    modulation tensor (or scalar) that can be applied per postsynaptic neuron.
    
    For example, in a two-neuron output layer, you can supply a vector such as
    [positive_const, negative_const] to reinforce the correct action and depress the other.
    """
    def __init__(
        self,
        connection,
        nu=None,
        reduction=None,
        weight_decay=0.0,
        **kwargs
    ) -> None:
        super().__init__(connection=connection, nu=nu, reduction=reduction, weight_decay=weight_decay, **kwargs)

        # This implementation supports connections that have a weight matrix (w).
        if hasattr(connection, "w"):
            self.update = self._connection_update
        else:
            raise NotImplementedError("This learning rule is not supported for this Connection type.")

        self.tc_plus = torch.tensor(kwargs.get("tc_plus", 20.0))
        self.tc_minus = torch.tensor(kwargs.get("tc_minus", 20.0))
        self.tc_e_trace = torch.tensor(kwargs.get("tc_e_trace", 25.0))

    def _connection_update(self, step=0, **kwargs) -> None:
        """
        Custom MSTDPET update with per-neuron reward modulation.
        
        Keyword arguments:
        
        :param Union[float, torch.Tensor] reward: Reward signal. If a tensor is passed (e.g.,
            a vector of length equal to the number of target neurons), then each column of the
            weight matrix is modulated differently.
        :param float a_plus: Learning rate component for the pre-synaptic trace.
        :param float a_minus: Learning rate component for the post-synaptic trace.
        """
        # Initialize eligibility variables on the first call or when step == 0.
        if not hasattr(self, "p_plus") or step == 0:
            self.p_plus = torch.zeros(self.connection.source.n)
        if not hasattr(self, "p_minus") or step == 0:
            self.p_minus = torch.zeros(self.connection.target.n)
        if not hasattr(self, "eligibility") or step == 0:
            self.eligibility = torch.zeros(*self.connection.w.shape)
        if not hasattr(self, "eligibility_trace") or step == 0:
            self.eligibility_trace = torch.zeros(*self.connection.w.shape)

        # Get pre- and post-synaptic spikes as 1D float tensors.
        source_s = self.connection.source.s.reshape(-1).float()
        target_s = self.connection.target.s.reshape(-1).float()

        # Parse keyword arguments.
        reward = kwargs["reward"]
        a_plus = torch.tensor(kwargs.get("a_plus", 1.0))
        a_minus = torch.tensor(kwargs.get("a_minus", -1.0))

        # Update eligibility trace.
        self.eligibility_trace *= torch.exp(-self.connection.dt / self.tc_e_trace)
        self.eligibility_trace += self.eligibility / self.tc_e_trace

        # Prepare reward modulation.
        if isinstance(reward, torch.Tensor) and reward.dim() > 0:
            # Expect reward to be of shape (target_n,) --> reshape to (1, target_n) for broadcasting.
            reward_mod = reward.view(1, -1)
        else:
            reward_mod = reward

        # Compute weight update.
        update = self.nu[0] * self.connection.dt * self.eligibility_trace * reward_mod
        self.connection.w += update

        # Update pre- and post-synaptic traces.
        self.p_plus *= torch.exp(-self.connection.dt / self.tc_plus)
        self.p_plus += a_plus * source_s
        self.p_minus *= torch.exp(-self.connection.dt / self.tc_minus)
        self.p_minus += a_minus * target_s

        # Compute point eligibility.
        self.eligibility = torch.ger(self.p_plus, target_s) + torch.ger(source_s, self.p_minus)

        # Optionally call parent update.
        super().update()

class TDLTP(LearningRule):
    # language=rst
    """
    Temporal-difference long-term potentiation rule. Adapted from
    `(Fremaux 2013)`_.
    """

    def __init__(
        self,
        connection: AbstractConnection,
        nu: Optional[Union[float, Sequence[float]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        **kwargs
    ) -> None:
        # language=rst
        """
        Constructor for ``TDLTP`` learning rule.

        :param connection: An ``AbstractConnection`` object whose weights the
            ``TDLTP`` learning rule will modify.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events,
            respectively.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param weight_decay: Constant multiple to decay weights by on each iteration.

        Keyword arguments:

        :param float tc_plus: Time constant for pre-synaptic firing trace.
        :param float tc_minus: Time constant for post-synaptic firing trace.
        :param float tc_e_trace: Time constant for the eligibility trace.
        """
        super().__init__(
            connection=connection,
            nu=nu,
            reduction=reduction,
            weight_decay=weight_decay,
            **kwargs
        )

        if isinstance(connection, (Connection, LocalConnection)):
            self.update = self._connection_update
        else:
            raise NotImplementedError(
                "This learning rule is not supported for this Connection type."
            )

        self.eps_0 = torch.tensor(kwargs.get("eps_0", 150.0))
        self.tau_m = torch.tensor(kwargs.get("tau_m", 20.0))
        self.tau_s = torch.tensor(kwargs.get("tau_s", 5.0))
        self.tau_r = torch.tensor(kwargs.get("tau_r", 4e3))
        self.v_k = torch.tensor(kwargs.get("v_k", 50.0))
        self.tau_k = torch.tensor(kwargs.get("tau_k", 200.0))

    def _connection_update(self, td_error=None, step=0, **kwargs) -> None:
        # language=rst
        """
        TDLTP learning rule for ``Connection`` subclass of ``AbstractConnection``
        class.

        Keyword arguments:

        :param Union[float, torch.Tensor] reward: TD-error signal from reinforcement
            learning task 
        """
        # Initialize eligibility traces
        if not hasattr(self, "ic_eps_plus_trace") or step==0:
            self.ic_eps_plus_trace = torch.zeros(*((torch.transpose(self.connection.w, 0, 1)).shape))
        if not hasattr(self, "ic_eps_minus_trace") or step==0:
            self.ic_eps_minus_trace = torch.zeros(*((torch.transpose(self.connection.w, 0, 1)).shape))
        if not hasattr(self, "ic_kappa_plus_trace") or step==0:
            self.ic_kappa_plus_trace = torch.zeros(*((torch.transpose(self.connection.w, 0, 1)).shape))
        if not hasattr(self, "ic_kappa_minus_trace") or step==0:
            self.ic_kappa_minus_trace = torch.zeros(*((torch.transpose(self.connection.w, 0, 1)).shape))

        # Reshape pre- and post-synaptic spikes.
        source_s = (self.source.s).reshape(-1).float()
        target_s = (self.target.s).reshape(-1).float()

        # Parse keyword arguments.
        td_error = kwargs["reward"]

        
        self.ic_eps_plus_trace *= np.exp(-self.connection.dt / self.tau_m)
        self.ic_eps_minus_trace *= np.exp(-self.connection.dt / self.tau_s)
        
        if source_s.any():
            self.ic_eps_plus_trace[:, source_s.bool()] += \
                        self.eps_0 / (self.tau_m - self.tau_s)
            self.ic_eps_minus_trace[:, source_s.bool()] += \
                self.eps_0 / (self.tau_m - self.tau_s)
        
        self.ic_kappa_plus_trace *= np.exp(-self.connection.dt / self.tau_k)
        self.ic_kappa_minus_trace *= np.exp(-self.connection.dt / self.v_k)
        
        if target_s.any():
            self.ic_kappa_plus_trace[target_s.bool(), :] += 1 / ((self.tau_k - self.v_k)*self.tau_r)
            self.ic_kappa_minus_trace[target_s.bool(), :] += 1 / ((self.tau_k - self.v_k)*self.tau_r)
            
            self.ic_eps_plus_trace[target_s.bool(), :] = 0
            self.ic_eps_minus_trace[target_s.bool(), :] = 0

        outer_conv_kappa = (self.ic_eps_plus_trace - self.ic_eps_minus_trace) * (self.ic_kappa_plus_trace - self.ic_kappa_minus_trace)
        
        update = self.nu[0] * self.connection.dt * td_error * torch.transpose(outer_conv_kappa, 0, 1)
        
        self.connection.w += update

        super().update()


class Rmax(LearningRule):
    # language=rst
    """
    Reward-modulated learning rule derived from reward maximization principles. Adapted
    from `(Vasilaki et al., 2009)
    <https://intranet.physio.unibe.ch/Publikationen/Dokumente/Vasilaki2009PloSComputBio_1.pdf>`_.
    """

    def __init__(
        self,
        connection: AbstractConnection,
        nu: Optional[Union[float, Sequence[float]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        **kwargs
    ) -> None:
        # language=rst
        """
        Constructor for ``R-max`` learning rule.

        :param connection: An ``AbstractConnection`` object whose weights the ``R-max``
            learning rule will modify.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events,
            respectively.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param weight_decay: Constant multiple to decay weights by on each iteration.

        Keyword arguments:

        :param float tc_c: Time constant for balancing naive Hebbian and policy gradient
            learning.
        :param float tc_e_trace: Time constant for the eligibility trace.
        """
        super().__init__(
            connection=connection,
            nu=nu,
            reduction=reduction,
            weight_decay=weight_decay,
            **kwargs
        )

        # Trace is needed for computing epsilon.
        assert (
            self.source.traces and self.source.traces_additive
        ), "Pre-synaptic nodes must use additive spike traces."

        # Derivation of R-max depends on stochastic SRM neurons!
        assert isinstance(
            self.target, SRM0Nodes
        ), "R-max needs stochastically firing neurons, use SRM0Nodes."

        if isinstance(connection, (Connection, LocalConnection)):
            self.update = self._connection_update
        else:
            raise NotImplementedError(
                "This learning rule is not supported for this Connection type."
            )

        self.tc_c = torch.tensor(
            kwargs.get("tc_c", 5.0)
        )  # 0 for pure naive Hebbian, inf for pure policy gradient.
        self.tc_e_trace = torch.tensor(kwargs.get("tc_e_trace", 25.0))

    def _connection_update(self, **kwargs) -> None:
        # language=rst
        """
        R-max learning rule for ``Connection`` subclass of ``AbstractConnection`` class.

        Keyword arguments:

        :param Union[float, torch.Tensor] reward: Reward signal from reinforcement
            learning task.
        """
        # Initialize eligibility trace.
        if not hasattr(self, "eligibility_trace"):
            self.eligibility_trace = torch.zeros(*self.connection.w.shape)

        # Reshape variables.
        target_s = self.target.s.view(-1).float()
        target_s_prob = self.target.s_prob.view(-1)
        source_x = self.source.x.view(-1)

        # Parse keyword arguments.
        reward = kwargs["reward"]

        # New eligibility trace.
        self.eligibility_trace *= 1 - self.connection.dt / self.tc_e_trace
        self.eligibility_trace += (
            target_s
            - (target_s_prob / (1.0 + self.tc_c / self.connection.dt * target_s_prob))
        ) * source_x[:, None]

        # Compute weight update.
        self.connection.w += self.nu[0] * reward * self.eligibility_trace

        super().update()
