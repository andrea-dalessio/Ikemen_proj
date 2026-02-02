import time
import yaml
import json
import os
import subprocess
import socket
import struct
import numpy as np
from pathlib import Path
from gymnasium import spaces

parentPath = Path(__file__).resolve().parent
configsPath = parentPath / 'configs.yaml'


with open(configsPath, 'r') as configsFile:
    CONFIGS = yaml.safe_load(configsFile)

    # actionStruct = {
    #     "p1_move": move, 
    #     "p1_btn": btn, 
    #     "p2_move": "",  # Player 2 fermo
    #     "p2_btn": "", 
    #     "reset": False
    #     "end": False
    # }

class IkemenEnvironment:
    host = CONFIGS['env']['host']
    def __init__(self, training_mode:str, port:int, instance:int=0, headless=False):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.max_retries = CONFIGS['env']['max_retries']
        self.connected = False
        self.previousState = None
        self.port = port
        self.log = open(f"{os.getcwd()}/logs/log_{self.port}.txt", 'w')
        self.instance = instance
        self.cycle_number = 0
        self.headless = headless
        self.keyMap = {
            'U': 'up',
            'D': 'down',
            'L': 'left',
            'R': 'right',
            'UL': 'up left',    # O "7" (Numpad)
            'UR': 'up right',   # O "9"
            'DL': 'down left',  # O "1"
            'DR': 'down right', # O "3"
            '-': ''             # Neutro
        }        
        
        self.actionMapHit = {
            0: "-",
            1: "a",
            2: "b",
            3: "x",
            4: "y",
            5: "ab",
            6: "xy"
        }
        
        self.move_intent = {
            0: "-",
            1: "F",
            2: "B",
            3: "U",
            4: "D",
            5: "UF",
            6: "UB",
            7: "DF",
            8: "DB"
        }        
        
        self.action_space = (len(self.move_intent), len(self.actionMapHit))
        self.time_limit = CONFIGS['env']['trunk_ticks']
        self.state_space = (19,)
        self.observation_space = (CONFIGS['env']['channel_number'], CONFIGS['env']['window_height'], CONFIGS['env']['window_width'])
        if training_mode == 'teacher':
            self.needFrame = False
        elif training_mode == 'student':
            self.needFrame = True
        else:
            raise ValueError(f"[{self.instance}] Invalid training mode. Choose 'teacher' or 'student'.")
        
        # Here's the subprocess constructor :)
        self.game_process = None
        
        # Track this to set the "t0" for each round
        self.round_start_tick = 0

    def wait_for_match_start(self, timeout=30):
        """
        Handshake protocol
        """
        print(f"[{self.instance}] Syncing with game...")
        start_time = time.time()
        
        if self.socket is None:
            raise Exception(f"[{self.instance}] Socket missing")
        
        while time.time() - start_time < timeout:
            try:
                # 1. Invia un comando "tutto fermo" per svegliare il server
                # Nota: usa gli indici 0 (Nessun movimento, Nessun attacco)
                # Adatta gli indici se 0 non è "Nothing" nella tua mappa
                self.executeAction((0, 0), (0, 0)) 
                
                # 2. Prova a leggere lo stato
                # Usiamo un timeout breve sul socket se possibile, o ci affidiamo al try/except
                self.socket.settimeout(1.0) 
                state, frame = self.recieve()
                self.socket.settimeout(None) # Rimuovi timeout
                
                if state is not None:
                    if state.get('p1_hp', 0) > 0:
                        print(f"[{self.instance}] Sync OK!")
                        return state, frame
                    
            except (ConnectionError, struct.error, socket.timeout):
                # Se fallisce, aspetta un po' e riprova
                time.sleep(0.5)
            except Exception as e:
                print(f"[{self.instance}] Unexpected error during sync: {e}")
                time.sleep(1)
                
        raise TimeoutError(f"[{self.instance}] Il gioco non ha risposto entro il tempo limite.")

    def connect(self):
        # 1. Kill all previous sockets
        if self.socket is not None:
            self.disconnect()

        # 2. Errno 106 (Transport endpoint is already connected) workaround
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        # Optional: Timeout to avoid infinite blocking in connect
        self.socket.settimeout(5.0) 

        print(f"[{self.instance}] Opening on {self.host}:{self.port}")
        for i in range(self.max_retries):
            try:
                self.socket.connect((self.host, self.port))
                self.connected = True
                self.socket.settimeout(None) # Remove timeout for normal operation
                return
            except (ConnectionError, socket.timeout, OSError) as e:
                print(f"[{self.instance}] Failed connection [{i+1}/{self.max_retries}]: {e}")
                time.sleep(2)
        
        # If all attempts fail, clean up
        self.socket = None 
        raise ConnectionError(f"[{self.instance}] Failed connection after retries")   
    
    # Here you launch the game!
    def launch_game(self):
        game_path = parentPath.parent / "game/Ikemen_GO_Linux"
        if not os.path.exists(game_path):
            raise FileNotFoundError(f"[{self.instance}] Game executable not found at {game_path}")
        
        portNumber = str(self.port) # Adjust as needed  
        # launch_args = ['xvfb-run', '-a', str(game_path), '-p1', 'kfm', '-p2', 'kfm', '-ai', '0', '-port', portNumber] # Set to infinite time per round
        env = os.environ.copy()
        if self.headless:
            print(f"[{self.instance}] Headless mode")
            env["SDL_AUDIODRIVER"] = "dummy"
            launch_args = ["xvfb-run","-a", "-s", "-screen 0 1280x720x24", str(game_path), '-p1', 'kfm', '-p2', 'kfm', '-ai', '0', '-port', portNumber]
        else:
            print(f"[{self.instance}] Headed mode")
            launch_args = [str(game_path), '-p1', 'kfm', '-p2', 'kfm', '-ai', '0', '-port', portNumber] # Set to infinite time per round
        print(f"[{self.instance}] Launching IkemenGO...")
        
        try:
            self.game_process = subprocess.Popen(launch_args, cwd=os.path.dirname(game_path), stdout=self.log, stderr=self.log, env=env)
            print(f"[{self.instance}] Game launched, waiting for server...")
            time.sleep(1)
        except Exception as e:
            print(f"[{self.instance}] Failed to launch game: {e}")
            raise e
    
    def disconnect(self):
        if self.socket is not None:
            try:
                self.send({'reset':True})
                self.socket.shutdown(socket.SHUT_RDWR)
            except (OSError, Exception):
                pass
            
            try:
                self.socket.close()
            except (OSError, Exception):
                pass

            self.socket = None
        
        self.connected = False

    def close_game(self):
        self.disconnect()
        if self.game_process is not None:
            print(f"[{self.instance}] Closing Ikemen GO...")
            self.game_process.terminate()
            try:
                self.game_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.game_process.kill()
                print(f"[{self.instance}] Game process killed because it crashed.")
            self.game_process = None
    
    def recieveHelper(self, n:int):
        if self.socket is None:
            raise ConnectionError(f'[{self.instance}] No socket')
        elif not self.connected:
            raise ConnectionError(f'[{self.instance}] Not connected')
        
        buf = b''
        while len(buf) < n:
            chunk = self.socket.recv(n - len(buf))
            if not chunk:
                raise ConnectionError
            buf += chunk
        return buf
    
    def send(self, data:dict):
        if self.game_process is not None:
            if self.game_process.poll() is not None:
                self.connected = False
                self.socket = None
                raise ConnectionError(f"[{self.instance}] Game Process Died unexpectedy.")

        if self.socket is None:
            raise ConnectionError(f'[{self.instance}] No socket')
        elif not self.connected:
            raise ConnectionError(f'[{self.instance}] Not connected')
        
        try:
            payload = json.dumps(data).encode('utf-8')
            header = struct.pack('>I', len(payload))
            self.socket.sendall(header + payload)
        except BrokenPipeError:
            self.connected = False
            raise ConnectionError(f"[{self.instance}] Broken Pipe during send (Server closed connection).")
    
    def rewardCompute(self, state):
        current_tick = state.get('tick', 0)
        relative_tick = current_tick - self.round_start_tick
        self.cycle_number += 1
        
        # Warm-up (ignora primi frame)
        if relative_tick < 60:
            return 0.0, False

        terminated = False
        truncated = False
        reward = 0.0
        
        MAX_LIFE = float(state.get('p1_life_max', 1000))
        p1_hp = state.get('p1_hp', 0)
        p2_hp = state.get('p2_hp', 0)
        
        # Fine Match
        if p1_hp <= 0 or p2_hp <= 0:
            print(f"[{self.instance}] Fighter KO")
            terminated = True
            
        elif self.cycle_number > self.time_limit:
            print(f"[{self.instance}] Time over")
            truncated = True
        
        done = terminated or truncated
        
        if self.previousState is not None:
            # --- 1. HEALTH REWARD (Invariato, buono) ---
            # Premia molto il danno fatto, punisce metà il danno subito.
            # Questo incoraggia il "trading" aggressivo.
            diff_p2 = self.previousState['p2_hp'] - p2_hp
            diff_p1 = self.previousState['p1_hp'] - p1_hp
            if diff_p2 < 0: diff_p2 = 0
            if diff_p1 < 0: diff_p1 = 0
            
            reward += (diff_p2 / MAX_LIFE) * 3.0
            reward -= (diff_p1 / MAX_LIFE) * 4.0   
            
            # --- 2. NUOVO: CLOSING IN REWARD (Invece della Distance Penalty) ---
            # Se ti avvicini, ti do un biscottino. Se ti allontani, niente (o piccolo malus).
            # Questo insegna a "cacciare" l'avversario invece di scappare.
            prev_dist = abs(self.previousState.get('p1_x',0) - self.previousState.get('p2_x',0))
            curr_dist = abs(state.get('p1_x',0) - state.get('p2_x',0))
            
            # Tieni la pressione. Tension up!
            if curr_dist < prev_dist:
                reward += 0.002 
            
            elif curr_dist > prev_dist:
                retreat_pen = 0.005 # Se ti arretri, negative penalty!
                p1_x = state.get('p1_x',0)
                p1_facing = state.get('p1_facing',1)
                if ((p1_x < -900) and (p1_facing == 1)) or ((p1_x > 900) and (p1_facing == -1)):  # Backed up into corner penalty
                    retreat_pen *= 4.0
                
                if ((p1_x < -900) and (p1_facing == -1)) or ((p1_x > 900) and (p1_facing == 1)):  # Pressuring into corner bonus
                    retreat_pen *= 0.2
                
                if p1_hp > p2_hp:
                    retreat_pen *= 1.5  # Più punizione se sei in vantaggio
                    
                reward -= retreat_pen
            
            animno = state.get('p1_anim_no', 0)
            if 200 <= animno <= 800 and curr_dist > 150.0:
                reward -= 0.005
                
            
            # --- 3. DYNAMIC SLOW PLAY PENALTY ---
            FIGHT_RANGE = 180.0
            if curr_dist > FIGHT_RANGE:
                reward -= 0.002
            else:
                reward -= 0.0005  # Penalità minore se sei in range di combattimento
            
            # --- 4. WIN/LOSS MASSICCI ---
        if done:
            # Calcoliamo quanto è durato il match per il log
            match_len = relative_tick 
            
            # CASO 1: VITTORIA P1 (Teacher)
            if p1_hp > 0 and p2_hp <= 0:
                print(f"[{self.instance}] 🏆 WIN  | HP: {p1_hp} vs {p2_hp} | Duration: {match_len}/{self.time_limit} ticks")
                reward += 5.0 
                
            # CASO 2: SCONFITTA P1 (Vittoria Opponent)
            elif p1_hp <= 0 and p2_hp > 0:
                print(f"[{self.instance}] 💀 LOSS | HP: {p1_hp} vs {p2_hp} | Duration: {match_len}/{self.time_limit} ticks")
                reward -= 2.0 
                
            # CASO 3: DOPPIO KO (Pareggio)
            elif (p1_hp <= 0 and p2_hp <= 0) or truncated:
                print(f"[{self.instance}] 🤝 DRAW | HP: {p1_hp} vs {p2_hp} | Duration: {match_len}/{self.time_limit} ticks")
                # Un pareggio è meglio di una sconfitta, ma peggio di una vittoria
                reward -= 1.0
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
        
        # Now using "time elapsed" feat instead of "time remaining"
        raw_tick = state.get('tick', 0)
        current_tick = raw_tick - self.round_start_tick
        MAX_DURATION = 20000.0
        if current_tick < 0: current_tick = 0  # Safety check
        time_norm = current_tick / MAX_DURATION
        time_feat = np.clip(time_norm, 0.0, 1.0)
        
        prev_p1_x = self.previousState.get('p1_x', p1_x) if self.previousState else p1_x
        prev_p2_x = self.previousState.get('p2_x', p2_x) if self.previousState else p2_x
        prev_p1_y = self.previousState.get('p1_y', p1_y) if self.previousState else p1_y
        prev_p2_y = self.previousState.get('p2_y', p2_y) if self.previousState else p2_y
        
        # Velocities (normalized)
        p1_dx = (p1_x - prev_p1_x) / 20.0  # Assuming max speed of 20 units/frame (estimated, fine-tune as needed)
        p2_dx = (p2_x - prev_p2_x) / 20.0
        p1_dy = (p1_y - prev_p1_y) / 20.0
        p2_dy = (p2_y - prev_p2_y) / 20.0
        
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
            state.get('p1_power', 0) / 3000.0,
            state.get('p2_power', 0) / 3000.0,
            self.normalize_anim_smart(state.get('p1_anim_no', 0)),
            self.normalize_anim_smart(state.get('p2_anim_no', 0)),
            time_feat,
            p1_dx, p1_dy, p2_dx, p2_dy,
            state.get('p1_y', 0) / -200.0,   # Normalized position on max stage height
            state.get('p2_y', 0) / -200.0  # Normalized position on max stage height            
            ], 
            dtype=np.float32
        )
        
        return state_vector
        
    def recieve(self): # Editing module to return frame only if needed. Also correcting bugs and restructuring state vector
        json_size = struct.unpack('>I', self.recieveHelper(4))[0]
        raw = json.loads(self.recieveHelper(json_size))
        
        nextState = raw['state']
        
        
        img_size = struct.unpack('>I', self.recieveHelper(4))[0]
        img = self.recieveHelper(img_size)
        
        frame = None
        if self.needFrame:
            w = raw["frame_w"]
            h = raw["frame_h"]
            frame = np.frombuffer(img, dtype=np.uint8).reshape((h, w, 4))
        flip = nextState.get("p1_facing", 1) == -1
        if flip:
            frame = np.flip(frame, axis=1).copy()
        
        return nextState, frame

    # Rewritten because action detection was faulty (for movements)
    def executeAction(self, actionP1, actionP2):
        if not self.connected:
            return

        # Facing directions
        p1_facing = 1
        p2_facing = -1
        
        if self.previousState is not None:
            p1_facing = self.previousState.get('p1_facing', 1)
            p2_facing = self.previousState.get('p2_facing', -1)

        # Helper function to get physical key from intent
        def get_physical_key(intent_idx, facing):
            intent = self.move_intent[int(intent_idx)]
            
            # Up or NO-MOVE
            if intent in ["-", "U", "D"]:
                return self.keyMap[intent]
            
            # Relative logic F/B
            is_forward = 'F' in intent
            is_back = 'B' in intent
            
            side_key = ""
            if is_forward:
                side_key = 'R' if facing == 1 else 'L'
            elif is_back:
                side_key = 'L' if facing == 1 else 'R'
                
            # Combination for diagonals (e.g., UF -> U + side_key)
            final_key = ""
            if 'U' in intent: final_key = 'U' + side_key # Es. UR
            elif 'D' in intent: final_key = 'D' + side_key # Es. DR
            else: final_key = side_key # Es. R
            
            return self.keyMap.get(final_key, "")

        # Get physical keys for both players
        move_str_p1 = get_physical_key(actionP1[0], p1_facing)
        move_str_p2 = get_physical_key(actionP2[0], p2_facing)
        
        btn_str_p1 = self.actionMapHit[int(actionP1[1])]
        btn_str_p2 = self.actionMapHit[int(actionP2[1])]
        
        # Build and send payload
        payload = {
            "p1_move": move_str_p1,
            "p1_btn": btn_str_p1,
            "p2_move": move_str_p2,
            "p2_btn": btn_str_p2,
            "reset": False,
            "end": False
        }
        
        self.send(payload)

    def reset(self):
        if self.socket is None or not self.connected:
            raise ConnectionError(f"[{self.instance}] Tried resetting while there is no socket")
        try:
            self.send({'reset':True})
            time.sleep(0.5)
            self.socket.setblocking(False)
            try:
                while self.socket.recv(4096):
                    pass
            except BlockingIOError:
                pass
            self.socket.setblocking(True)
            new_raw_state, new_frame = self.wait_for_match_start()
            self.cycle_number = 0
            self.round_start_tick = new_raw_state.get('tick', 0)
            self.previousState = new_raw_state
            return new_raw_state, new_frame
        except Exception as e:
            print(f"[{self.instance}] Connection Error: {e}")
            raise e

    def start(self):
        self.launch_game()
        try:
            self.connect()
        except ConnectionError as ex:
            print(f"[{self.instance}] Could not connect: {ex}")
            
    # Adding a sync protocol...
    def sync_step(self):
        if self.socket is None:
            return None

        try:
            # Ping server with a no-op action
            self.executeAction((0, 0), (0, 0))
            
            # Quick listen (0.05s timeout)
            self.socket.settimeout(0.05) 
            state, frame = self.recieve()
            self.socket.settimeout(None) # Restore blocking
            
            if state is not None and state.get('p1_hp', 0) > 0:
                return state, frame
                
        except (ConnectionError, struct.error, socket.timeout):
            pass
        except Exception as e:
            print(f"[{self.instance}] Sync step error: {e}")
            
        return None