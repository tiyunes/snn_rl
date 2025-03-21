Использованиие подхода Mixed actor-critic для решения задачи **Acrobot** [mixed_ac_acrobot.py]

Actor представляет собой спайковую нейронную сеть (вход сети - неспайковый, скрытый и выходной слои - спайковые), обучение происходит на основе PGCN (Policy Gradien Coagent Network) [Aenugu, 2020]
Critic - спайковая нейронная сеть, обучение происходит на основе правила TD-LTP [Fremaux, Gerstner, 2013]

График награды* в процессе обучения:

![train_rewards_acrobot](https://github.com/tiyunes/snn_rl/assets/79756733/609f8ea4-e976-4323-952b-27029b4e2d79)


График награды на 70 тестовых итерациях:

![test_rewards_acrobot](https://github.com/tiyunes/snn_rl/assets/79756733/02fe8f11-07b6-4abd-93c2-ae30198bc884)



Среднее значение награды на 70 тестовых итерациях: -169.93

* в процессе обучения каждый неуспешный шаг вычиталось 10 единиц награды, за успешное завершение начислялось 50 единиц награды. В процессе тестирования каждый шаг вносит вклад -1 единиц награды

Использованиие подхода Mixed actor-critic для решения задачи **Cartpole** [mixed_ac_cartpole.py]

График награды в процессе обучения:

![overall_train_reward_w_running](https://github.com/tiyunes/snn_rl/assets/79756733/477da3e3-1b35-42c5-b34d-288aad969932)
