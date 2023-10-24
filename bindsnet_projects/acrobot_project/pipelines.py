"""
==============================================================================.

pipelines.py

@author: atenagm

==============================================================================.
"""

import os
import datetime
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from bindsnet.pipeline.environment_pipeline import EnvironmentPipeline

from agents import ObserverAgent, ExpertAgent


class ToMPipeline(EnvironmentPipeline):
    """
    Abstracts the interaction between agents in the environment.

    Parameters
    ----------
    observer_agent : ObserverAgent
        The oberserver agent in the environment.
    expert_agent : ExpertAgent
        The expert agent in the environment.
    encoding : callable, optional
        The observation encoder function that returns a dict of torch tensors.
        The default is None.

    Keyword Arguments
    -----------------
    num_episodes : int
        Number of episodes to train for. The default is 100.
    render_interval : int
        Interval to render the environment.
    reward_delay : int
        How many iterations to delay delivery of reward.
    time : int
        Time for which to run the network.
    overlay_input : int
        Overlay the last X previous input.
    representation_time : int
        The time to represent the expert's action to Premotor of the observer.
        The default is -1.
    log_writer : bool
        Whether to create text log files of weights and spikes (For debug purpose).
        The default is False.

    """

    def __init__(
            self,
            observer_agent: ObserverAgent,
            expert_agent: ExpertAgent,
            encoding: callable = None,
            **kwargs,
    ) -> None:

        assert (observer_agent.environment == expert_agent.environment), \
            "Observer and Expert must be located in the same environment."
        assert (observer_agent.device == expert_agent.device), \
            "Observer and Expert objects must be on same device."

        self.device = observer_agent.device
        self.encoding = encoding
        self.observer_agent = observer_agent
        self.observer_agent.build_network()
        kwargs["output"] = kwargs.get("output", "PM")

        super().__init__(
            observer_agent.network,
            observer_agent.environment,
            encoding=encoding,
            allow_gpu=observer_agent.allow_gpu,
            **kwargs,
        )

        self.expert_agent = expert_agent

        self.representation_time = kwargs.get('representation_time', -1)

        self.keep_state = kwargs.get('keep_state', False)

        self.log_writer = kwargs.get('log_writer', False)

        self.plot_config = {
            "data_step": True,
        }

        self.test_rewards = []
        if self.log_writer:
            self.time_recorder = 0
            now = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M")
            self._log_path = f"./log-{now}"
            os.mkdir(self._log_path)

    def env_step(self, **kwargs) -> tuple:
        """
        Perform single step of the environment.

        Includes rendering, getting and performing the action, and
        accumulating/delaying rewards.

        KeywordArguments
        ----------------

        Returns
        -------
        obs : torch.Tensor
            The observation tensor.
        reward : float
            The reward value.
        done : bool
            Indicates if the the episode is terminated.
        info : dict
            The information dictionary for verbose.

        """
        # Render game.
        if (
            self.render_interval is not None
            and self.step_count % self.render_interval == 0
        ):
            self.env.render()

        # Choose action.
        if self.action_function is not None:
            self.action = self.action_function(episode=self.episode,
                                               num_episodes=self.num_episodes,
                                               **kwargs)

        # Run a step of the environment.
        self.action = int(self.action.item())
        obs, reward, done, info = self.env.step(self.action)

        # Set reward in case of delay.
        if self.reward_delay is not None:
            self.rewards = torch.tensor([reward, *self.rewards[1:]]).float()
            reward = self.rewards[-1]

        # Accumulate reward.
        self.accumulated_reward += reward

        info["accumulated_reward"] = self.accumulated_reward

        return obs, reward, done, info

    def _mirror(self) -> dict:
        out_n = self.network.layers[self.output].n
        n_action = self.env.action_space.n
        out_s = torch.zeros(self.time, out_n)
        out_s[self.representation_time, out_n//n_action * self.action:
             out_n//n_action * (self.action + 1)] = 1

        out_s = out_s.view(self.time, n_action, -1).bool().to(self.device)
        return {self.output: out_s}

    def step_(
            self,
            gym_batch: tuple,
            **kwargs
    ) -> None:
        """
        Run one step of the oberserver's network.

        Parameters
        ----------
        gym_batch : tuple[torch.Tensor, float, bool, dict]
            An OpenAI gym compatible tuple.

        Keyword Arguments
        -----------------
        log_path : str
            The path to save plots per step of each episode.

        Returns
        -------
        None

        """
        obs, reward, done, info = gym_batch
        # Encode the observation
        inputs = self.encoding(obs, self.time, **kwargs)

        # Clamp output to spike at `representation_time` based on expert's action.
        clamp = {}
        if self.network.learning:
            clamp = self._mirror()

        # Run the network
        self.network.run(inputs=inputs, clamp=clamp, time=self.time,
                         reward=reward, **kwargs)

        if kwargs.get("log_path") is not None and not self.observer_agent.network.learning:
            self._log_info(kwargs["log_path"], obs.squeeze(), inputs)

        if done:
            if self.network.learning and self.network.reward_fn is not None:
                self.network.reward_fn.update(
                    accumulated_reward=self.accumulated_reward,
                    steps=self.step_count,
                    **kwargs,
                )

            # Update reward list for plotting purposes.
            self.reward_list.append(self.accumulated_reward)

    def __get_init_state(self) -> tuple:
        obs = torch.Tensor(self.env.env.state).to(self.device)
        reward = 1.
        done = False
        info = {}
        return (obs, reward, done, info)

    def _train(self, training_act=False, **kwargs):
        while self.episode < self.num_episodes:
            if not training_act:
                self.observer_agent.network.train(True)

            # Expert acts in the environment.
            self.action_function = self.expert_agent.select_action

            # Initialize environment.
            self.reset_state_variables()
            prev_obs, prev_reward, prev_done, info = self.__get_init_state()

            new_done = False
            while not prev_done:
                if not self.keep_state:
                    self.network.reset_state_variables()
                else:
                    self.reset_monitors()

                prev_done = new_done
                # The result of expert's action.
                if not prev_done:
                    new_obs, new_reward, new_done, info = self.env_step(**kwargs)

                # The observer watches the result of expert's action and how
                # it modified the environment.
                self.step((prev_obs, prev_reward, prev_done, info), **kwargs)
                if self.log_writer:
                    self._save_simulation_info(**kwargs)

                prev_obs = new_obs
                prev_reward = new_reward

            print(
                f"Episode: {self.episode} - "
                f"accumulated reward: {self.accumulated_reward:.2f}"
            )

            self._test(training_act, **kwargs)
            self.episode += 1

    def _test(self, training_act, **kwargs) -> None:
        test_interval = kwargs.get("test_interval", None)
        num_tests = kwargs.get("num_tests", 1)
        log_count = 0

        if test_interval is not None:
            if (self.episode + 1) % test_interval == 0:
                for nt in range(num_tests):
                    self.act(training_act,
                             num=(nt + 1) + (log_count * 5), **kwargs)
                log_count += 1

    def observe_learn(self, **kwargs):
        self.observer_agent.network.train(True)
        self._train(**kwargs)

    def observe_act_learn(self, **kwargs):
        self.observer_agent.network.train(True)
        self._train(training_act=True, **kwargs)

    def train_by_observation(self, **kwargs) -> None:
        """
        Train observer agent's network by observing the expert.

        Keyword Arguments
        -----------------
        test_interval : int
            The interval of testing the network. The default is 1.
        num_tests : int
            The number of tests in each test interval. The default is 1.

        Returns
        -------
        None

        """
        self.observer_agent.network.train(True)

        test_interval = kwargs.get("test_interval", 1)
        num_tests = kwargs.get("num_tests", 1)
        log_count = 0
        while self.episode < self.num_episodes:
            self.observer_agent.network.train(True)
            # Expert acts in the environment.
            self.action_function = self.expert_agent.select_action
            self.reset_state_variables()

            prev_obs = torch.Tensor(self.env.env.state).to(self.observer_agent.device)
            prev_reward = 1.
            prev_done = False
            info = {}

            new_done = False
            while not prev_done:
                self.network.reset_state_variables()

                prev_done = new_done
                # The result of expert's action.
                if not prev_done:
                    new_obs, new_reward, new_done, info = self.env_step(**kwargs)

                # The observer watches the result of expert's action and how
                # it modified the environment.
                self.step((prev_obs, prev_reward, prev_done, info), **kwargs)
                if self.log_writer:
                    self._save_simulation_info(**kwargs)

                prev_obs = new_obs
                prev_reward = new_reward

            print(
                f"Episode: {self.episode} - "
                f"accumulated reward: {self.accumulated_reward:.2f}"
            )

            if test_interval is not None:
                if (self.episode + 1) % test_interval == 0:
                    for nt in range(num_tests):
                        self.act(num=(nt + 1) + (log_count * 5), **kwargs)
                    log_count += 1

            self.episode += 1

    def act(self, training_act=False, **kwargs) -> None:
        """
        Test the observer agent in the environment.

        Returns
        -------
        None

        """
        self.observer_agent.network.train(training_act)

        self.reset_state_variables()

        self.action_function = self.observer_agent.select_action
        obs = torch.Tensor(self.env.env.state).to(self.observer_agent.device)
        self.step((obs, 1.0, False, {}), **kwargs)

        if self.log_writer:
            self._save_simulation_info(**kwargs)

        done = False
        while not done:
            # The result of observer's action.
            obs, reward, done, info = self.env_step(**kwargs)

            self.network.reset_state_variables()
            self.step((obs, reward, done, info), **kwargs)

            if self.log_writer:
                self._save_simulation_info(**kwargs)

        self.test_rewards.append(self.reward_list.pop())
        print("Test - accumulated reward:", self.accumulated_reward)

    def reset_state_variables(self) -> None:
        super().reset_state_variables()
        if not self.network.learning and self.observer_agent.method == 'first_spike':
            self.observer_agent.random_counter = 0

    def reset_env(self) -> None:
        self.env.reset()
        self.accumulated_reward = 0.0
        self.step_count = 0
        self.overlay_start = True
        self.action = torch.tensor(-1)
        self.last_action = torch.tensor(-1)
        self.action_counter = 0

    def reset_monitors(self) -> None:
        for monitor in self.network.monitors:
            self.network.monitors[monitor].reset_state_variables()

    def _save_simulation_info(self, **kwargs):
        spikes = torch.cat([
            self.network.monitors["S2"].get("s").squeeze().view(self.time, -1),
            self.network.monitors["PM"].get("s").squeeze(),
        ], dim=1).nonzero()

        if self.observer_agent.network.learning:
            with open(f"{self._log_path}/train_spikes{self.episode}.txt", 'a') as tr:
                for spike in spikes:
                    tr.write(f"{spike[0] + self.time_recorder} {spike[1]}\n")
            with open(f"{self._log_path}/train_weights{self.episode}.txt", 'a') as tr:
                w = self.network.monitors["S2-PM"].get("w")
                for wt in range(len(w)):
                    tr.write(f"{self.time_recorder + wt} {w[wt]}\n")
        else:
            num = kwargs["num"]
            with open(f"{self._log_path}/test_spikes{num}.txt", 'a') as tr:
                for spike in spikes:
                    tr.write(f"{spike[0] + self.time_recorder} {spike[1]}\n")

        self.time_recorder += self.time

    def _log_info(self, path, obs, encoded_input):
        # TODO change it later for inputs of shape n*m
        plt.ioff()
        fig, axes = plt.subplots(len(encoded_input.keys()), 2)
        for idx, k in enumerate(encoded_input.keys()):
            ss = encoded_input[k].squeeze().to("cpu")
            if len(ss.shape) == 2:
                ss = ss.nonzero()
                axes[idx, 0].scatter(ss[:, 0], ss[:, 1])
                axes[idx, 0].set(xlim=[-1, self.time + 1],
                                 ylim=[-1, encoded_input[k].shape[-1]])
                axes[idx, 0].set_title(k)
            else:
                # TODO reconsider
                for i in range(ss.shape[1]):
                    s = ss[:, i, :].nonzero()
                    axes[idx * i, 0].scatter(s[:, 0], s[:, 1])
                    plt.xlim([-1, self.time + 1])
                    plt.ylim([-1, ss.shape[-1]])

        v = self.network.monitors["PM"].get("v").squeeze().to("cpu")
        axes[0, 1].plot(v[:, 0], c='r', label="0")
        axes[0, 1].plot(v[:, 1], c='b', label="1")
        axes[0, 1].set(ylim=[self.network.layers["PM"].rest.to("cpu"),
                             self.network.layers["PM"].thresh.to("cpu")])
        axes[0, 1].legend()

        s = self.network.monitors["PM"].get("s").squeeze().nonzero().to("cpu")
        axes[1, 1].scatter(s[:, 0], s[:, 1])
        axes[1, 1].set(xlim=[-1, self.time + 1], ylim=[-1, 2])
        fig.savefig(path + f"/{self.episode}_{len(self.test_rewards)}_"
                    f"{self.step_count}_{obs}_{self.action}.png")
        fig.clf()



class AcrobotPipeline(EnvironmentPipeline):
    """
    Abstracts the interaction between agents in the environment.

    Parameters
    ----------
    observer_agent : ObserverAgent
        The oberserver agent in the environment.
    expert_agent : ExpertAgent
        The expert agent in the environment.
    encoding : callable, optional
        The observation encoder function that returns a dict of torch tensors.
        The default is None.

    Keyword Arguments
    -----------------
    num_episodes : int
        Number of episodes to train for. The default is 100.
    render_interval : int
        Interval to render the environment.
    reward_delay : int
        How many iterations to delay delivery of reward.
    time : int
        Time for which to run the network.
    overlay_input : int
        Overlay the last X previous input.
    representation_time : int
        The time to represent the expert's action to Premotor of the observer.
        The default is -1.
    log_writer : bool
        Whether to create text log files of weights and spikes (For debug purpose).
        The default is False.

    """

    def __init__(
            self,
            observer_agent: ObserverAgent,
            expert_agent: ExpertAgent,
            encoding: callable = None,
            **kwargs,
    ) -> None:

        assert (observer_agent.environment == expert_agent.environment), \
            "Observer and Expert must be located in the same environment."
        assert (observer_agent.device == expert_agent.device), \
            "Observer and Expert objects must be on same device."

        self.device = observer_agent.device
        self.encoding = encoding
        self.observer_agent = observer_agent
        self.observer_agent.build_network()
        kwargs["output"] = kwargs.get("output", "PM")

        super().__init__(
            observer_agent.network,
            observer_agent.environment,
            encoding=encoding,
            allow_gpu=observer_agent.allow_gpu,
            **kwargs,
        )

        self.expert_agent = expert_agent

        self.representation_time = kwargs.get('representation_time', -1)

        self.keep_state = kwargs.get('keep_state', False)

        self.log_writer = kwargs.get('log_writer', False)

        self.plot_config = {
            "data_step": True,
        }

        self.test_rewards = []
        if self.log_writer:
            self.time_recorder = 0
            now = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M")
            self._log_path = f"./log-{now}"
            os.mkdir(self._log_path)

    def env_step(self, **kwargs) -> tuple:
        """
        Perform single step of the environment.

        Includes rendering, getting and performing the action, and
        accumulating/delaying rewards.

        KeywordArguments
        ----------------

        Returns
        -------
        obs : torch.Tensor
            The observation tensor.
        reward : float
            The reward value.
        done : bool
            Indicates if the the episode is terminated.
        info : dict
            The information dictionary for verbose.

        """
        # Render game.
        if (
            self.render_interval is not None
            and self.step_count % self.render_interval == 0
        ):
            self.env.render()

        # Choose action.
        if self.action_function is not None:
            self.action = self.action_function(episode=self.episode,
                                               num_episodes=self.num_episodes,
                                               **kwargs)

        # Run a step of the environment.
        if isinstance(self.action, list):
            self.action = self.action[0]

        obs, reward, done, info = self.env.step(self.action)

        # Set reward in case of delay.
        if self.reward_delay is not None:
            self.rewards = torch.tensor([reward, *self.rewards[1:]]).float()
            reward = self.rewards[-1]

        # Accumulate reward.
        self.accumulated_reward += reward

        info["accumulated_reward"] = self.accumulated_reward

        return obs, reward, done, info

    def _mirror(self) -> dict:
        out_n = self.network.layers[self.output].n
        n_action = self.env.action_space.n
        out_s = torch.zeros(self.time, out_n)
        out_s[self.representation_time, out_n//n_action * self.action:
             out_n//n_action * (self.action + 1)] = 1

        out_s = out_s.view(self.time, n_action, -1).bool().to(self.device)
        out_s_np = out_s.numpy()
        return {self.output: out_s}

    def step_(
            self,
            gym_batch: tuple,
            **kwargs
    ) -> None:
        """
        Run one step of the oberserver's network.

        Parameters
        ----------
        gym_batch : tuple[torch.Tensor, float, bool, dict]
            An OpenAI gym compatible tuple.

        Keyword Arguments
        -----------------
        log_path : str
            The path to save plots per step of each episode.

        Returns
        -------
        None

        """
        obs, reward, done, info = gym_batch
        # Encode the observation
        inputs = self.encoding(obs, self.time, **kwargs)

        # Clamp output to spike at `representation_time` based on expert's action.
        clamp = {}
        if self.network.learning:
            clamp = self._mirror()

        # Run the network
        self.network.run(inputs=inputs, clamp=clamp, time=self.time,
                         reward=reward, **kwargs)

        if kwargs.get("log_path") is not None and not self.observer_agent.network.learning:
            self._log_info(kwargs["log_path"], obs.squeeze(), inputs)

        if done:
            if self.network.learning and self.network.reward_fn is not None:
                self.network.reward_fn.update(
                    accumulated_reward=self.accumulated_reward,
                    steps=self.step_count,
                    **kwargs,
                )

            # Update reward list for plotting purposes.
            self.reward_list.append(self.accumulated_reward)

    def __get_init_state(self) -> tuple:
        state = torch.Tensor(self.env.env.state).to(self.device)
        cos1 = np.cos(state[0])
        sin1 = np.sin(state[0])
        cos2 = np.cos(state[1])
        sin2 = np.sin(state[1])
        obs = torch.tensor([cos1, sin1, cos2, sin2, state[2], state[3]])
        reward = 0.0
        done = False
        info = {}
        return (obs, reward, done, info)

    def _train(self, training_act=False, **kwargs):
        while self.episode < self.num_episodes:
            if not training_act:
                self.observer_agent.network.train(True)

            # Expert acts in the environment.
            self.action_function = self.expert_agent.select_action

            # Initialize environment.
            self.reset_state_variables()
            prev_obs, prev_reward, prev_done, info = self.__get_init_state()

            new_done = False
            while not prev_done:
                if not self.keep_state:
                    self.network.reset_state_variables()
                else:
                    self.reset_monitors()

                prev_done = new_done
                # The result of expert's action.
                if not prev_done:
                    new_obs, new_reward, new_done, info = self.env_step(**kwargs)

                # The observer watches the result of expert's action and how
                # it modified the environment.
                self.step((prev_obs, prev_reward, prev_done, info), **kwargs)
                if self.log_writer:
                    self._save_simulation_info(**kwargs)

                prev_obs = new_obs
                prev_reward = new_reward

            print(
                f"Episode: {self.episode} - "
                f"accumulated reward: {self.accumulated_reward:.2f}"
            )

            self._test(training_act, **kwargs)
            self.episode += 1

    def _test(self, training_act, **kwargs) -> None:
        test_interval = kwargs.get("test_interval", None)
        num_tests = kwargs.get("num_tests", 1)
        log_count = 0

        if test_interval is not None:
            if (self.episode + 1) % test_interval == 0:
                for nt in range(num_tests):
                    self.act(training_act,
                             num=(nt + 1) + (log_count * 5), **kwargs)
                log_count += 1

    def observe_learn(self, **kwargs):
        self.observer_agent.network.train(True)
        self._train(**kwargs)

    def observe_act_learn(self, **kwargs):
        self.observer_agent.network.train(True)
        self._train(training_act=True, **kwargs)

    def train_by_observation(self, **kwargs) -> None:
        """
        Train observer agent's network by observing the expert.

        Keyword Arguments
        -----------------
        test_interval : int
            The interval of testing the network. The default is 1.
        num_tests : int
            The number of tests in each test interval. The default is 1.

        Returns
        -------
        None

        """
        self.observer_agent.network.train(True)

        test_interval = kwargs.get("test_interval", 1)
        num_tests = kwargs.get("num_tests", 1)
        log_count = 0
        while self.episode < self.num_episodes:
            self.observer_agent.network.train(True)
            # Expert acts in the environment.
            self.action_function = self.expert_agent.select_action
            self.reset_state_variables()

            prev_obs = torch.Tensor(self.env.env.state).to(self.observer_agent.device)
            prev_reward = 1.
            prev_done = False
            info = {}

            new_done = False
            while not prev_done:
                self.network.reset_state_variables()

                prev_done = new_done
                # The result of expert's action.
                if not prev_done:
                    new_obs, new_reward, new_done, info = self.env_step(**kwargs)

                # The observer watches the result of expert's action and how
                # it modified the environment.
                self.step((prev_obs, prev_reward, prev_done, info), **kwargs)
                if self.log_writer:
                    self._save_simulation_info(**kwargs)

                prev_obs = new_obs
                prev_reward = new_reward

            print(
                f"Episode: {self.episode} - "
                f"accumulated reward: {self.accumulated_reward:.2f}"
            )

            if test_interval is not None:
                if (self.episode + 1) % test_interval == 0:
                    for nt in range(num_tests):
                        self.act(num=(nt + 1) + (log_count * 5), **kwargs)
                    log_count += 1

            self.episode += 1

    def act(self, training_act=False, **kwargs) -> None:
        """
        Test the observer agent in the environment.

        Returns
        -------
        None

        """
        self.observer_agent.network.train(training_act)

        self.reset_state_variables()

        self.action_function = self.observer_agent.select_action
        state = torch.Tensor(self.env.env.state).to(self.observer_agent.device)
        cos1 = np.cos(state[0])
        sin1 = np.sin(state[0])
        cos2 = np.cos(state[1])
        sin2 = np.sin(state[1])
        obs = torch.tensor([cos1, sin1, cos2, sin2, state[2], state[3]])
        self.step((obs, 1.0, False, {}), **kwargs)

        if self.log_writer:
            self._save_simulation_info(**kwargs)

        done = False
        while not done:
            # The result of observer's action.
            obs, reward, done, info = self.env_step(**kwargs)

            self.network.reset_state_variables()
            self.step((obs, reward, done, info), **kwargs)

            if self.log_writer:
                self._save_simulation_info(**kwargs)

        self.test_rewards.append(self.reward_list.pop())
        print("Test - accumulated reward:", self.accumulated_reward)

    def reset_state_variables(self) -> None:
        super().reset_state_variables()
        if not self.network.learning and self.observer_agent.method == 'first_spike':
            self.observer_agent.random_counter = 0

    def reset_env(self) -> None:
        self.env.reset()
        self.accumulated_reward = 0.0
        self.step_count = 0
        self.overlay_start = True
        self.action = torch.tensor(-1)
        self.last_action = torch.tensor(-1)
        self.action_counter = 0

    def reset_monitors(self) -> None:
        for monitor in self.network.monitors:
            self.network.monitors[monitor].reset_state_variables()

    def _save_simulation_info(self, **kwargs):
        spikes = torch.cat([
            self.network.monitors["S2"].get("s").squeeze().view(self.time, -1),
            self.network.monitors["PM"].get("s").squeeze(),
        ], dim=1).nonzero()

        if self.observer_agent.network.learning:
            with open(f"{self._log_path}/train_spikes{self.episode}.txt", 'a') as tr:
                for spike in spikes:
                    tr.write(f"{spike[0] + self.time_recorder} {spike[1]}\n")
            with open(f"{self._log_path}/train_weights{self.episode}.txt", 'a') as tr:
                w = self.network.monitors["S2-PM"].get("w")
                for wt in range(len(w)):
                    tr.write(f"{self.time_recorder + wt} {w[wt]}\n")
        else:
            num = kwargs["num"]
            with open(f"{self._log_path}/test_spikes{num}.txt", 'a') as tr:
                for spike in spikes:
                    tr.write(f"{spike[0] + self.time_recorder} {spike[1]}\n")

        self.time_recorder += self.time

    def _log_info(self, path, obs, encoded_input):
        # TODO change it later for inputs of shape n*m
        plt.ioff()
        fig, axes = plt.subplots(len(encoded_input.keys()), 2)
        for idx, k in enumerate(encoded_input.keys()):
            ss = encoded_input[k].squeeze().to("cpu")
            if len(ss.shape) == 2:
                ss = ss.nonzero()
                axes[idx, 0].scatter(ss[:, 0], ss[:, 1])
                axes[idx, 0].set(xlim=[-1, self.time + 1],
                                 ylim=[-1, encoded_input[k].shape[-1]])
                axes[idx, 0].set_title(k)
            else:
                # TODO reconsider
                for i in range(ss.shape[1]):
                    s = ss[:, i, :].nonzero()
                    axes[idx * i, 0].scatter(s[:, 0], s[:, 1])
                    plt.xlim([-1, self.time + 1])
                    plt.ylim([-1, ss.shape[-1]])

        v = self.network.monitors["PM"].get("v").squeeze().to("cpu")
        axes[0, 1].plot(v[:, 0], c='r', label="0")
        axes[0, 1].plot(v[:, 1], c='b', label="1")
        axes[0, 1].set(ylim=[self.network.layers["PM"].rest.to("cpu"),
                             self.network.layers["PM"].thresh.to("cpu")])
        axes[0, 1].legend()

        s = self.network.monitors["PM"].get("s").squeeze().nonzero().to("cpu")
        axes[1, 1].scatter(s[:, 0], s[:, 1])
        axes[1, 1].set(xlim=[-1, self.time + 1], ylim=[-1, 2])
        fig.savefig(path + f"/{self.episode}_{len(self.test_rewards)}_"
                    f"{self.step_count}_{obs}_{self.action}.png")
        fig.clf()