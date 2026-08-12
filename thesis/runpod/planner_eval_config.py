"""Контрольный конфиг: тот же JSON-бенчмарк, но задачу решает скриптовый планировщик.

Зачем. Прогон π0-FAST даёт прогрессию 0.133 и ноль успехов. Само по себе это не
говорит, исправен ли стенд: ноль одинаково объясняется и слабой моделью, и сломанным
захватом. Планировщик получает координаты объекта от симулятора и от зрения не зависит,
поэтому его результат на том же бенчмарке отделяет одно от другого:

    планировщик берёт предмет  -> робот, гриппер, физика и засчитывание захвата исправны,
                                  низкие числа π0 относятся к модели;
    планировщик не берёт       -> дело в стенде, и числа π0 measure ничего не значат.

Наследуемся от JsonBenchmarkEvalConfig, как предписывает molmo_spaces/evaluation/README.md:
внешние конфиги должны поставить только robot_config и policy_config, всё остальное
приходит из бенчмарка.

Планировщик берётся обычный (pick_and_place_planner_policy), не curobo-вариант:
в нём нет ни одного упоминания curobo, поэтому он работает в окружении .[mujoco],
ровно как на evox2, где curobo тоже не установлен.

Запуск:
    python -m molmo_spaces.evaluation.eval_main \
        --benchmark_dir <...> --output_dir <...> --max_episodes 3 --no_wandb \
        planner_eval_config:PlannerBenchmarkEvalConfig
"""

from molmo_spaces.configs.policy_configs import PickAndPlacePlannerPolicyConfig
from molmo_spaces.configs.robot_configs import FrankaRobotConfig
from molmo_spaces.evaluation.configs.evaluation_configs import JsonBenchmarkEvalConfig


class PlannerBenchmarkEvalConfig(JsonBenchmarkEvalConfig):
    robot_config: FrankaRobotConfig = FrankaRobotConfig()
    policy_config: PickAndPlacePlannerPolicyConfig = PickAndPlacePlannerPolicyConfig()

    # Планировщик выдаёт по одному действию на шаг, чанками не оперирует, поэтому
    # частота управления берётся штатная для генерации данных, а не 15 Гц от DROID.
    policy_dt_ms: float = 100.0
    end_on_success: bool = True

    @property
    def tag(self) -> str:
        return "planner_benchmark_eval"

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        # Шум действий выключен: контрольный прогон должен быть максимально
        # благоприятным для планировщика, иначе его неудача ничего не докажет.
        self.robot_config.action_noise_config.enabled = False
