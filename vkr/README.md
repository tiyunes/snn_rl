# Трёхфакторное обучение спайковых нейронных сетей (actor-critic, TD-STDP)

Код и модели ВКР: спайковый actor (скрытый слой обучен по STDP, выходной — по TD-STDP) и спайковый / неспайковый critic для CartPole, Acrobot, Mountain Car.

```
pip install -r requirements.txt
python nonspiking_critic/acrobot/test.py
```

`bindsnet/` — модифицированная BindsNET 0.2.7 (AGPL-3.0, см. `bindsnet/LICENSE`). Скрипты используют именно её.

## 1. Скрытый слой actor-а (STDP)

| Папка | Обучение | Чекпойнт |
|---|---|---|
| `hidden_layers/acrobot_20n` | `train_hidden.py` | `hidden_layer.pt` |
| `hidden_layers/acrobot_100n` | `train_hidden.py` | `hidden_layer.pt` |
| `hidden_layers/cartpole_500n` | `train_hidden.py` | `hidden_layer.pt` |
| `hidden_layers/mountain_car_500n` | `train_hidden.py` | `hidden_layer.pt` |
| `hidden_layers/mountain_car_1000n` | `train_hidden.py` | `hidden_layer.pt` |

Датасеты: `data/`. Для CartPole нужен `data/states_for_scale.json` (датасет из Vlasov et al., 2023, в репозиторий не включён).

## 2. Неспайковый critic

| Папка | Скрытый слой | Обучение | Итоговый чекпойнт | Кривая обучения | Тест |
|---|---|---|---|---|---|
| `nonspiking_critic/acrobot` | acrobot_20n | `train.py` | `actor.pt` | `training_log.pt` | `test.py` |
| `nonspiking_critic/cartpole` | cartpole_500n | `train.py` | `actor.pt` | `training_log.pt` | `test.py` |
| `nonspiking_critic/mountain_car` | mountain_car_500n | `train.py` | `actor.pt` + `critic.npy` | `training_log.json` | `test.py` |

## 3. Спайковый critic

| Папка | Скрытый слой | Обучение | Итоговый чекпойнт | Кривая обучения | Тест |
|---|---|---|---|---|---|
| `spiking_critic/acrobot` | acrobot_100n | `train.py` | `actor.pt` + `critic.pt` | `training_log.pt` | `test.py` |
| `spiking_critic/cartpole` | cartpole_500n | `train.py` | `checkpoint.pt` | тот же файл | `test.py` |
| `spiking_critic/mountain_car` | mountain_car_1000n | `train.py` | `checkpoint.pt` | `training_log.pt` | `test.py` |

Итоговые чекпойнты содержат весь актор, включая обученный выходной слой H→Y.

Результаты новых запусков сохраняются в `runs/` рядом со скриптом.
