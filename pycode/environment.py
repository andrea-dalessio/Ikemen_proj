import time
import yaml
import json
import socket
import struct
import argparse
import numpy as np
from pathlib import Path

configsPath = Path(__file__).resolve().parent / 'configs.yaml'


with open(configsPath, 'r') as configsFile:
    CONFIGS = yaml.safe_load(configsFile)

class IkemenEnvironment:
    host = CONFIGS['env']['host']
    port = CONFIGS['env']['port']
    
    # actionStruct = {
    #     "p1_move": move, 
    #     "p1_btn": btn, 
    #     "p2_move": "",  # Player 2 fermo
    #     "p2_btn": "", 
    #     "reset": False
    # }
    
    actionMapHit = {
        0: "-",
        1: "a",
        2: "b",
        3: "x",
        4: "y",
    }
    
    actionMapMove = {
        0: "-",
        1: "forward",
        2: "back",
        3: "up",
        4: "down",
    }
    
    def __init__(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.max_retries = CONFIGS['env']['max_retries']
        self.connected = False
        self.previousState = None

    def connect(self):
        if self.socket is None:
            raise ConnectionError('No socket')
        elif self.connected:
            return
        
        for i in range(self.max_retries):
            try:
                self.socket.connect((self.host, self.port))
                self.connected = True
                return
            except ConnectionError as e:
                print(f"Failed connection [{i+1}/{self.max_retries}]: {e}")
                time.sleep(2)
        raise ConnectionError("Failed connection")    
    
    def disconnect(self):
        if not self.socket is None and self.connected:
            self.socket.close()
            self.socket = None
    
    def recieveHelper(self, n:int):
        if self.socket is None:
            raise ConnectionError('No socket')
        elif not self.connected:
            raise ConnectionError('Not connected')
        
        buf = b''
        while len(buf) < n:
            chunk = self.socket.recv(n - len(buf))
            if not chunk:
                raise ConnectionError
            buf += chunk
        return buf
    
    def send(self, data:dict):
        if self.socket is None:
            raise ConnectionError('No socket')
        elif not self.connected:
            raise ConnectionError('Not connected')
        
        payload = json.dumps(data).encode('utf-8')
        header = struct.pack('>I', len(payload))
        self.socket.sendall(header + payload)
    

    def evaluateState(self, state):
        done = False
        reward = 0
        
        # Condizione di fine episodio (Done)
        if state['p1_hp'] <= 0 or state['p2_hp'] <= 0:
            done = True
        
        # Ricompensa basata sui danni
        if self.previousState is not None:
            # Ricompensa principale: P1 guadagna per il danno fatto, perde per il danno subito
            p1_damage_delta = self.previousState['p2_hp'] - state['p2_hp']
            p2_damage_delta = self.previousState['p1_hp'] - state['p1_hp']
            
            # Ricompensa istantanea
            reward += p1_damage_delta 
            reward -= p2_damage_delta 

            # Ricompensa per la vittoria/sconfitta (terminale)
            if done:
                if state['p1_hp'] > 0:
                    reward += 100
                else:
                    reward -= 100

        # Assicura di avere uno stato precedente per il calcolo differenziale
        if self.previousState is None:
            reward = 0

        return reward, done

    def normalize_anim_smart(self, anim_no):
        if anim_no >= 5000:
            compact_anim = 153 + (anim_no - 5000)
        else:
            compact_anim = anim_no
        NEW_MAX = 453.0 
        compact_anim = min(compact_anim, NEW_MAX)
        
        return compact_anim / NEW_MAX

    def normalizeState(self, state):
        """ Transforms the raw state into a normalized state vector for the teacher model. Good so you don't have to normalize it later!"""
        p1_x = state.get('p1_x', 0)
        p2_x = state.get('p2_x', 0)
        p1_y = state.get('p1_y', 0)
        p2_y = state.get('p2_y', 0)

        # Using normalized hp values to max values and distance between players
        state_vector = np.array([
            state.get('p1_hp', 0) / state.get('p1_life_max', 1000),
            state.get('p2_hp', 0) / state.get('p2_life_max', 1000),
            (p2_x - p1_x) / 1000.0,  # Assuming max distance of 1000 units. Safe constant used to keep into account zoom out feature.
            (p2_y - p1_y) / 600.0,  # Assuming max vertical distance of 600 units (enemy can be launched higher than upper screen bound)
            (p1_x + 1000.0) / 2000.0,  # Normalized position on max stahge width (2000 units)
            (p2_x + 1000.0) / 2000.0,
            state.get('p1_facing', 1),  # Facing direction (1 or -1)
            state.get('p2_facing', 1),
            state.get('p1_power', 0) / 100.0,
            state.get('p2_power', 0) / 100.0,
            self.normalize_anim_smart(state.get('p1_anim_no', 0)),
            self.normalize_anim_smart(state.get('p2_anim_no', 0))
        ], dtype=np.float32) # TODO : Add timer
        
        return state_vector
        
    
    def recieve(self, needFrame:bool=False): # Editing module to return frame only if needed. Also correcting bugs and restructuring state vector
        json_size = struct.unpack('>I', self.recieveHelper(4))[0]
        raw = json.loads(self.recieveHelper(json_size))
        
        nextState = raw['state']
        
        img_size = struct.unpack('>I', self.recieveHelper(4))[0]
        img = self.recieveHelper(img_size)
        
        frame = None
        if needFrame:
            w = raw["frame_w"]
            h = raw["frame_h"]
            frame = np.frombuffer(img, dtype=np.uint8).reshape((h, w, 4))
        
        return nextState, frame



    def executeAction(self, actionP1:tuple[int,int], actionP2:tuple[int,int]):
        nextMove = {
            "p1_move": self.actionMapMove[actionP1[0]], 
            "p1_btn": self.actionMapHit[actionP1[1]], 
            "p2_move": self.actionMapMove[actionP2[0]], 
            "p2_btn": self.actionMapHit[actionP2[1]], 
            "reset": False
        }
        self.send(nextMove)

    def reset(self):
        if self.socket is None or not self.connected:
            return
        try:
            chunk = self.socket.recv(0)
            if not chunk:
                return 
        except ConnectionError as e:
            print(f"Connection Error: {e}")
            return
        self.send({'reset':True})
        


#-------------------------------|Test section|-------------------------------------------------

parser = argparse.ArgumentParser(description="Ikemen Environment Test")
parser.add_argument('--test', action = 'store_true', help='Run environment test')

if __name__ == '__main__' and parser.parse_args().test:
    env = IkemenEnvironment()
    env.connect()
    cnt = 0
    while cnt < 10000:
        try: 
            json_size = struct.unpack('>I', env.recieveHelper(4))[0]
            raw = json.loads(env.recieveHelper(json_size))
            print(raw)
            
            state = raw['state']
            
            img_size = struct.unpack('>I', env.recieveHelper(4))[0]
            print(f"IMG Size: {img_size}")
            img = env.recieveHelper(img_size)
            
            w = raw["frame_w"]
            h = raw["frame_h"]
            print(f"W: {w}, H: {h}")
            
            rawImage = np.frombuffer(img, dtype=np.uint8)
            
            print(f"Image shape{rawImage.shape}")
            
            frame = rawImage.reshape((h, w, 4))
            
            nextMove = {
            "p1_move": env.actionMapMove[1], 
            "p1_btn": env.actionMapHit[0], 
            "p2_move": env.actionMapMove[1], 
            "p2_btn": env.actionMapHit[0], 
            "reset": False
            }

            env.send(nextMove)
            
            cnt += 1
            
        except KeyboardInterrupt:
            print("Manual interruption")
            env.disconnect()
            break
    if env.connected:
        env.disconnect()
        print("Test over")