import time
import yaml
import socket
from pathlib import Path

configsPath = Path(__file__).resolve().parent / 'configs.yaml'


with open(configsPath, 'r') as configsFile:
    CONFIGS = yaml.safe_load(configsFile)


class IkemenEnvironment:
    host = CONFIGS['env']['host']
    port = CONFIGS['env']['port']
    
    def __init__(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.max_retries = CONFIGS['env']['max_retries']
        self.connected = False
    
    def connect(self):
        for _ in range(self.max_retries):
            try:
                self.socket.connect((self.host, self.port))
            except ConnectionError as e:
                print(f"Failed connection: {e}")
                return
        
        self.connected = True
    
    def step(self, action):
        pass

    def reset(self):
        pass
    
    def disconnect(self):
        pass