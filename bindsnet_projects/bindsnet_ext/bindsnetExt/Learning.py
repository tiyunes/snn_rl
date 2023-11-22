from abc import ABC
from typing import Union, Optional, Sequence
import warnings

import torch
import numpy as np

from bindsnet.network.topology import (
    AbstractConnection,
    Connection,
    Conv2dConnection,
    LocalConnection,
)

from bindsnet.learning import LearningRule

class ppxPostPre(LearningRule):
    # language=rst


    def __init__(
        self,
        connection: AbstractConnection,
        nu: Optional[Union[float, Sequence[float]]] = (0.011, 0.316),
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        **kwargs
    ) -> None:
        
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
        self.beta_post = -5.969
        self.beta_pre = 2.213
        self.gamma_post = 0.146
        self.gamma_pre = 0.0318
        
        if isinstance(connection, (Connection, LocalConnection)):
            self.update = self._connection_update
        else:
            raise NotImplementedError(
                "This learning rule is not supported for this Connection type."
            )
            
        
    def A_plus(self, w, wMin, wMax, beta_pre): #pre-synaptic
        return torch.exp(-beta_pre * (wMax - w) / (wMax - wMin))

    def A_minus(self, w, wMin, wMax, beta_post): #post-synaptic
        return torch.exp(-beta_post * (w - wMin) / (wMax - wMin))

    def x(self, dt, tNorm, gamma_pre): #pre-synaptic
        return torch.abs(dt/tNorm)*torch.exp(-gamma_pre * torch.square((dt/tNorm)))

    def y(self, dt, tNorm, gamma_post): #post-synaptic
        return torch.abs(dt/tNorm)*torch.exp(-gamma_post * torch.square((dt/tNorm))) 
    
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
        
        wMin = self.wmin
        wMax = self.wmax

        # Negative update.
        if self.nu[0]:
            
            dt = torch.log(target_x)
            
            #dt_i = log(exp(-dt*i / tc_trace)) -- getting time differences from traces
            Y = self.y(dt, 1, self.gamma_post)
            #Y = y(t) * sum[DiracDelta(t - t_n)] -- same expression adapted for numeric calculations
            A_neg = self.A_minus(self.connection.w, wMin, wMax, self.beta_post)
            #A+ -- linear weight dependency
            
            prod = self.nu[0] * Y            
            #prod.view(batch_size, -1).unsqueeze(1)
            
            update -= self.reduction(torch.bmm(source_s, prod), dim=0) * A_neg
            

        # Positive update.
        if self.nu[1]:
            dt = torch.log(source_x)
            
            #dt_i = log(exp(-dt*i / tc_trace)) -- getting time differences from traces
            X = self.x(dt, 1, self.gamma_pre)
            #X = x(t) * sum[DiracDelta(t - t_n)] -- same expression adapted for numeric calculations
            A_pos = self.A_plus(self.connection.w, wMin, wMax, self.beta_pre)
            #A- -- linear weight dependency
            
            prod = self.nu[1] * X 
            
            
            update += self.reduction(torch.bmm(prod, target_s), dim=0) * A_pos
            
        
            
        update[torch.isnan(update)] = 0.
        

        self.connection.w += update

        super().update()
    
class ppxRSTDP(LearningRule):
    # language=rst


    def __init__(
        self,
        connection: AbstractConnection,
        nu: Optional[Union[float, Sequence[float]]] = (0.011, 0.316),
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        reward: float = 1.0,
        **kwargs
    ) -> None:
        
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
        self.beta_post = -5.969
        self.beta_pre = 2.213
        self.gamma_post = 0.146
        self.gamma_pre = 0.0318
        self.reward = reward
        
        if isinstance(connection, (Connection, LocalConnection)):
            self.update = self._connection_update
        else:
            raise NotImplementedError(
                "This learning rule is not supported for this Connection type."
            )
            
        
    def A_plus(self, w, wMin, wMax, beta_pre): #pre-synaptic
        return torch.exp(-beta_pre * (wMax - w) / (wMax - wMin))

    def A_minus(self, w, wMin, wMax, beta_post): #post-synaptic
        return torch.exp(-beta_post * (w - wMin) / (wMax - wMin))

    def x(self, dt, tNorm, gamma_pre): #pre-synaptic
        return torch.abs(dt/tNorm)*torch.exp(-gamma_pre * torch.square((dt/tNorm)))

    def y(self, dt, tNorm, gamma_post): #post-synaptic
        return torch.abs(dt/tNorm)*torch.exp(-gamma_post * torch.square((dt/tNorm))) 
    
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
        self.reward = kwargs.get("reward", 1)
        print(self.reward)
        
        update = 0
        
        wMin = self.wmin
        wMax = self.wmax

        # Negative update.
        if self.nu[0]:
            
            dt = torch.log(target_x)
            
            #dt_i = log(exp(-dt*i / tc_trace)) -- getting time differences from traces
            Y = self.y(dt, 1, self.gamma_post)
            #Y = y(t) * sum[DiracDelta(t - t_n)] -- same expression adapted for numeric calculations
            A_neg = self.A_minus(self.connection.w, wMin, wMax, self.beta_post)
            #A+ -- linear weight dependency
            
            prod = self.nu[0] * Y            
            #prod.view(batch_size, -1).unsqueeze(1)
            
            update -= self. reward * self.reduction(torch.bmm(source_s, prod), dim=0) * A_neg
            

        # Positive update.
        if self.nu[1]:
            dt = torch.log(source_x)
            
            #dt_i = log(exp(-dt*i / tc_trace)) -- getting time differences from traces
            X = self.x(dt, 1, self.gamma_pre)
            #X = x(t) * sum[DiracDelta(t - t_n)] -- same expression adapted for numeric calculations
            A_pos = self.A_plus(self.connection.w, wMin, wMax, self.beta_pre)
            #A- -- linear weight dependency
            
            prod = self.nu[1] * X 
            
            
            update += self. reward * self.reduction(torch.bmm(prod, target_s), dim=0) * A_pos
            
        
            
        update[torch.isnan(update)] = 0.
        

        self.connection.w += update

        super().update()
        
class ncPostPre(LearningRule):
    # language=rst


    def __init__(
        self,
        connection: AbstractConnection,
        nu: Optional[Union[float, Sequence[float]]] = (0.047, 0.074),
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        **kwargs
    ) -> None:
        
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
        self.mu_post = -22.300
        self.mu_pre = 26.700
        self.tau_post = -10.800
        self.tau_pre = 9.300
        
        if isinstance(connection, (Connection, LocalConnection)):
            self.update = self._connection_update
        elif isinstance(connection, Conv2dConnection):
            self.update = self._conv2d_connection_update
        else:
            raise NotImplementedError(
                "This learning rule is not supported for this Connection type."
            )
            
        
    def A_plus(self, w, wMin, wMax): #pre-synaptic
        return w/wMax

    def A_minus(self, w, wMin, wMax): #post-synaptic
        return w/wMax

    def x(self, dt, mu_pre, tau_pre): #pre-synaptic
        return torch.ones(dt.shape).to(dt.device) + torch.tanh(-(dt - mu_pre) / tau_pre)

    def y(self, dt, mu_post, tau_post): #post-synaptic
        return torch.ones(dt.shape).to(dt.device) - torch.tanh((dt - mu_post) / tau_post)

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
        
        wMin = self.wmin
        wMax = self.wmax

        # Negative update.
        if self.nu[0]:
            
            dt = torch.log(target_x) * self.target.tc_trace
            
            #dt_i = log(exp(-dt*i / tc_trace)) -- getting time differences from traces
            Y = self.y(dt, self.mu_post, self.tau_post)
            #Y = y(t) * sum[DiracDelta(t - t_n)] -- same expression adapted for numeric calculations
            A_neg = self.A_minus(self.connection.w, wMin, wMax)
            #A+ -- linear weight dependency
            
            prod = self.nu[0] * Y            
            #prod.view(batch_size, -1).unsqueeze(1)
            
            update -= self.reduction(torch.bmm(source_s, prod), dim=0) * A_neg
            

        # Positive update.
        if self.nu[1]:
            dt =  - torch.log(source_x) * self.source.tc_trace
            
            #dt_i = log(exp(-dt*i / tc_trace)) -- getting time differences from traces
            X = self.x(dt, self.mu_pre, self.tau_pre)
            #X = x(t) * sum[DiracDelta(t - t_n)] -- same expression adapted for numeric calculations
            A_pos = self.A_plus(self.connection.w, wMin, wMax)
            #A- -- linear weight dependency
            
            prod = self.nu[1] * X 
            
            
            update += self.reduction(torch.bmm(prod, target_s), dim=0) * A_pos
            
        
            
        update[torch.isnan(update)] = 0.
        

        self.connection.w += update

        super().update()