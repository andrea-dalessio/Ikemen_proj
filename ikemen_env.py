import socket
import json
import time

class IkemenEnv:
    """
    Interfaccia Ambiente per IKEMEN GO tramite socket TCP.
    Simula le funzionalità di base di un ambiente OpenAI Gym (reset, step).
    """

    HOST = '127.0.0.1'
    PORT = 8080
    BUFFER_SIZE = 4096
    ACTION_MAP = {
        0: {"p1_move": "", "p1_btn": ""},  # Neutro (Stare fermo)
        1: {"p1_move": "F", "p1_btn": ""},  # Avanti
        2: {"p1_move": "B", "p1_btn": ""},  # Indietro
        3: {"p1_move": "D", "p1_btn": ""},  # Basso (Parata, accovacciato)
        4: {"p1_move": "", "p1_btn": "a"},  # Pugno A
        5: {"p1_move": "", "p1_btn": "b"},  # Calcio B
        6: {"p1_move": "F", "p1_btn": "a"},  # Salto in avanti (o mossa complessa F+A)
    }

    def __init__(self, max_retries=10):
        self.sock = None
        self.max_retries = max_retries
        self.state = None 

    def connect(self):
        """Tenta di connettersi al server IKEMEN GO."""
        for attempt in range(self.max_retries):
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.connect((self.HOST, self.PORT))
                print("✅ Connessione TCP riuscita a IKEMEN GO.")
                return True
            except ConnectionRefusedError:
                print(f"Tentativo {attempt + 1}/{self.max_retries}: in attesa del server Go...")
                time.sleep(2)
        
        print("❌ Impossibile connettersi a IKEMEN GO. Assicurati che il gioco sia in esecuzione.")
        return False

    def receive_state(self):
        """Riceve lo stato di gioco JSON da IKEMEN."""
        data = b''
        try:
            self.sock.settimeout(1.0) 
            while True:
                chunk = self.sock.recv(self.BUFFER_SIZE)
                if not chunk:
                    print("⚠️ Connessione chiusa dal server Go.")
                    self.sock.close()
                    self.sock = None
                    return None
                data += chunk
                
                try:
                    state = json.loads(data.decode('utf-8'))
                    return state
                except json.JSONDecodeError:
                    continue 

        except socket.timeout:
            print("❗ Timeout ricezione stato. Potrebbe esserci un errore di sincronizzazione.")
            return None
        except Exception as e:
            print(f"Errore ricezione dati: {e}")
            return None

    def send_action(self, action_data):
        """Invia l'azione JSON al server IKEMEN."""
        try:
            msg = json.dumps(action_data).encode('utf-8')
            self.sock.sendall(msg)
            return True
        except Exception as e:
            print(f"Errore invio azione: {e}")
            return False

    def reset(self):
        """
        Invia il comando di Reset a IKEMEN e riceve il primo stato del nuovo round.
        """
        if not self.sock:
            self.connect()
            if not self.sock:
                raise ConnectionError("Impossibile resettare: connessione non stabilita.")

        # 1. Invia il comando di Reset (True)
        reset_action = {"p1_move": "", "p1_btn": "", "reset": True}
        if not self.send_action(reset_action):
            raise ConnectionError("Fallito l'invio del comando di reset.")
        
        # 2. Riceve il primo stato del nuovo episodio (Il server Go è resettato e aspetta)
        new_state = self.receive_state()
        if new_state is None:
            raise ConnectionError("Nessuna risposta dal server dopo il reset.")
            
        self.state = new_state
        print(f"🔄 Episodio resettato. Tick iniziale: {self.state['tick']}")
        return self.state

    def step(self, action_index: int):
        """
        Esegue un'azione nel gioco e restituisce (stato, ricompensa, done, info).
        
        Parametri:
            action_index (int): Indice dell'azione da ACTION_MAP.
        """
        if not self.sock:
            raise ConnectionError("Ambiente non connesso. Chiamare .connect() o .reset() prima.")

        # 1. Mappa l'indice numerico all'azione JSON
        action_payload = self.ACTION_MAP.get(action_index, self.ACTION_MAP[0])
        action_payload["reset"] = False # È un passo normale, non un reset
        
        # 2. Invia l'azione
        if not self.send_action(action_payload):
            raise ConnectionError("Fallito l'invio dell'azione.")

        # 3. Riceve il nuovo stato
        new_state = self.receive_state()
        if new_state is None:
            # Se la ricezione fallisce, trattalo come se fosse terminato
            return self.state, -100, True, {"error": "Connection lost"}

        # 4. Calcola Reward, Done, Info
        reward, done = self.calculate_reward_and_done(new_state)

        self.state = new_state
        return new_state, reward, done, {}

    def calculate_reward_and_done(self, new_state):
        """
        Logica centrale di ricompensa e fine episodio (DA PERSONALIZZARE).
        """
        done = False
        reward = 0
        
        # Condizione di fine episodio (Done)
        if new_state['p1_hp'] <= 0 or new_state['p2_hp'] <= 0:
            done = True
        
        # Ricompensa basata sui danni
        if self.state is not None:
            # Ricompensa principale: P1 guadagna per il danno fatto, perde per il danno subito
            p1_damage_delta = self.state['p2_hp'] - new_state['p2_hp']
            p2_damage_delta = self.state['p1_hp'] - new_state['p1_hp']
            
            # Ricompensa istantanea (es. 1 punto per HP tolto all'avversario)
            reward += p1_damage_delta 
            reward -= p2_damage_delta # Penalità per il danno subito

            # Ricompensa per la vittoria/sconfitta (terminale)
            if done:
                if new_state['p1_hp'] > 0:
                    reward += 100 # Ricompensa grande per la vittoria
                else:
                    reward -= 100 # Penalità grande per la sconfitta

        # Assicura di avere uno stato precedente per il calcolo differenziale
        if self.state is None:
            reward = 0

        return reward, done

    def close(self):
        """Chiude il socket."""
        if self.sock:
            self.sock.close()
            self.sock = None
            print("Connessione IKEMEN chiusa.")

# Esempio d'uso del modulo (Test)
if __name__ == "__main__":
    env = IkemenEnv()
    
    if env.connect():
        try:
            # 1. Inizia il primo episodio
            initial_state = env.reset()
            print(f"Stato Iniziale (Tick {initial_state['tick']}): P1 HP={initial_state['p1_hp']}")
            
            # 2. Esegui 100 passi di prova (Azioni casuali)
            steps = 0
            done = False
            while steps < 100 and not done:
                # Sostituisci questo con model.predict(state)
                action_to_take = steps % 7 # Azioni cicliche 0, 1, 2, ...
                
                state, reward, done, info = env.step(action_to_take)
                
                print(f"Step {steps}: Azione {action_to_take} -> Reward={reward:.2f}, Done={done}, HP={state['p1_hp']}")
                
                steps += 1
                time.sleep(0.01) # Rallenta un po' per l'osservazione
                
            # 3. Resetta se l'episodio è terminato per KO o Timeout RL
            if done:
                print("Episodio terminato, resetting...")
                env.reset()

        except Exception as e:
            print(f"Errore durante l'esecuzione del loop: {e}")
        finally:
            env.close()