from pathlib import Path
from .environment import IkemenEnvironment
from .models import TModelBase, SModelBase
from .superEnv import SuperEnvironment
import yaml

configsPath = Path(__file__).resolve().parent / 'configs.yaml'

with open(configsPath, 'r') as configsFile:
    CONFIGS = yaml.safe_load(configsFile)

class TeacherModel(models.TModelBase):
    def __init__(self, env, *args, **kwargs):
        super().__init__(env, configs=CONFIGS, *args, **kwargs)
        
class StudentModel(models.SModelBase):
    def __init__(self, env, *args, **kwargs):
        super().__init__(env, configs=CONFIGS, *args, **kwargs)
        