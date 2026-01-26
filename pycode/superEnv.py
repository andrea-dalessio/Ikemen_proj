import time
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
        print(f"Launching and connecting to {self.count} environments...")
        for e in self.envs:
            e.launch_game()
            time.sleep(2)
            
        print("Game initialization...")
        for e in self.envs:
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
            
    def wait_for_match_start(self, timeout=60):
        print("Parallel handshaking with environments...")
        start_time = time.time()
        results = [None] * self.count # Results from games
        synced_mask = [False] * self.count # Is the round over?
        
        while not all(synced_mask):
            if time.time() - start_time > timeout:
                raise TimeoutError("Global sync timed out.")

            for i, env in enumerate(self.envs):
                if synced_mask[i]:
                    continue
                
                # Single sync attempt
                res = env.sync_step()
                
                if res is not None:
                    results[i] = res
                    synced_mask[i] = True
                    print(f"[{i}] Synced!")
            
            # Small sleep to avoid maxing out the CPU in the while loop
            if not all(synced_mask):
                time.sleep(0.1)

        # Unpacking results
        states = [r[0] for r in results]
        frames = [r[1] for r in results]
        
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
        
    def hard_restart(self):
        print("!!! HARD RESTART TRIGGERED !!!")
        self.close_game()
        time.sleep(2) # Pulizia risorse OS
        self.start() # Rilancia e riconnette
        return self.wait_for_match_start()
    
    def normalizeState(self, state, index:int|None=None):
        if index is None:
            statesVector = []
            for i in range(self.count):
                statesNormalized = self.envs[i].normalizeState(state[i])
                statesVector.append(statesNormalized)
            return np.array(statesVector)
        else:
            statesNormalized = self.envs[index].normalizeState(state)
            return np.array(statesNormalized)