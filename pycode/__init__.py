from pathlib import Path
from .environment import IkemenEnvironment
from .models import TModelBase, SModelBase
import yaml

configsPath = Path(__file__).resolve().parent / 'configs.yaml'

with open(configsPath, 'r') as configsFile:
    CONFIGS = yaml.safe_load(configsFile)

class TeacherModel(models.TModelBase):
    def __init__(self, env,  device='cpu', *args, **kwargs):
        super().__init__(env, configs=CONFIGS, device=device, *args, **kwargs)
        
class StudentModel(models.SModelBase):
    def __init__(self, env,  device='cpu', *args, **kwargs):
        super().__init__(env, configs=CONFIGS, device=device, *args, **kwargs)
        