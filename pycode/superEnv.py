from .environment import IkemenEnvironment
import numpy as np
from pathlib import Path
import yaml


parentPath = Path(__file__).resolve().parent
configsPath = parentPath / 'configs.yaml'


with open(configsPath, 'r') as configsFile:
    CONFIGS = yaml.safe_load(configsFile)

class SuperEnvironment:
    def __init__(self, training_mode:str, environment_number:int=4) -> None:
        self.count = environment_number
        self.basePort = int(CONFIGS['env']['port'])
        self.envs:list[IkemenEnvironment] = []
        
        for i in range(self.count):
            env = IkemenEnvironment(training_mode, self.basePort+i, i)
            self.envs.append(env)
            
        if training_mode == 'teacher':
            self.needFrame = False
        elif training_mode == 'student':
            self.needFrame = True
    
    @property
    def observation_space(self):
        return self.envs[0].observation_space

    @property
    def action_space(self):
        return self.envs[0].action_space
    
    def start(self):
        for e in self.envs:
            e.launch_game()
            try:
                e.connect()
            except ConnectionError as ex:
                print(f"[{e.instance}] Could not connect: {ex}")
    
    def launch_game(self):
        for e in self.envs:
            e.launch_game()
    
    def connect(self):
        for e in self.envs:
            e.connect()
            
    def wait_for_match_start(self, timeout=30):
        states = []
        frames = []
        for e in self.envs:
            state, frame = e.wait_for_match_start(timeout=timeout)
            states.append(state)
            frames.append(frame)
        return states, np.array(frames)
    
    def disconnect(self):
        for e in self.envs:
            e.disconnect()
            
    def close_game(self):
        for e in self.envs:
            e.close_game()
            
    def executeAction(self, actionsP1, actionsP2):
        for i in range(self.count):
            self.envs[i].executeAction(actionP1=actionsP1[i], actionP2=actionsP2[i])
    
    def recieve(self):
        nextStates = []
        frames = []

        for e in self.envs:
            tmpState, tmpFrme = e.recieve()
            nextStates.append(tmpState)
            if self.needFrame:
                frames.append(tmpFrme)

        frames = np.array(frames)
        return nextStates, frames
    
    def rewardCompute(self, state):
        rew_vector = []
        dones = []
        for i in range(self.count):
            reward, done = self.envs[i].rewardCompute(state[i])
            rew_vector.append(reward)
            dones.append(done)
        return rew_vector, dones
    
    def reset(self, index:int|None=None):
        if index is None:
            frames = []
            states = []
            for e in self.envs:
                state, frame = e.reset()
                states.append(state)
                frames.append(frame)
            return states, np.array(frames)
        else:
            if index >= self.count:
                raise IndexError(f"Index out of order {index}(>={self.count})")
            state, frame = self.envs[index].reset()
            return state, np.array(frame)
    
    def normalizeState(self, state, index:int|None=None):
        if index is None:
            statesVector = []
            for i in range(self.count):
                statesNormalized = self.envs[i].normalizeState(state[i])
                statesVector.append(statesNormalized)
        else:
            statesNormalized = self.envs[index].normalizeState(state)    
        return np.array(statesVector)