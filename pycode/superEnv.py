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
    
    # Because the model compulsively sends attacks every frame (not very human-like and not well interpreted by the game),
    # we implement frame skipping to reduce the frequency of actions. (Here it executes the same action skipping 4 frames).
    # EDIT: Removing skipped frames but keeping the code as-is because the problem was in how the moving commands were parsed
    # by the game scripts).
    def step(self, actionsP1, actionsP2, skip=1):
        total_rewards = np.zeros(self.count, dtype=np.float32)
        final_dones = [False] * self.count
        current_states = [None] * self.count
        
        # Loop Frame Skipping
        for _ in range(skip):
            # 1. Sends action to all envs
            self.executeAction(actionsP1, actionsP2)
            
            # 2. Receives states (from all envs)
            # Note: recieve returns (states_list, frames_array)
            states, _ = self.recieve() 
            
            # 3. Process results for each environment
            for i in range(self.count):
                # If this env has already finished in a previous frame of the skip loop, ignore it
                if final_dones[i]:
                    continue
                
                # Calculate reward for the single frame
                step_reward, step_done = self.envs[i].rewardCompute(states[i])
                
                # Accumulate rewards and update states
                total_rewards[i] += step_reward
                current_states[i] = states[i]
                
                # IMPORTANT: Update the previousState of the specific env
                # Otherwise, in the next loop iteration, 'diff_hp' will be zero!
                self.envs[i].previousState = states[i]
                
                if step_done:
                    final_dones[i] = True
        
        # Returns the final state of the last frame (or the frame of match end)
        # The visual 'frame' (pixels) is not needed for the teacher, so we return None or empty
        return current_states, total_rewards, np.array(final_dones)    
    
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