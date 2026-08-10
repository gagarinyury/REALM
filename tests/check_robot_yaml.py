# Проверка определения робота по схеме — без симулятора и без загрузки сцены.
# Ошибки вида ConfigKeyError / несовпадение типов ловятся за секунду вместо четырёх минут.
import sys
from omegaconf import OmegaConf
from omnigibson.robots.definition_schema import RobotDefinition

path = sys.argv[1]
try:
    OmegaConf.merge(OmegaConf.structured(RobotDefinition), OmegaConf.load(path))
    print('OK: определение проходит схему')
except Exception as e:
    print('ОШИБКА:', type(e).__name__)
    print(str(e)[:600])
    sys.exit(1)
