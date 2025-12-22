import socket
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# --- CONFIGURAZIONE ---
HOST, PORT = '127.0.0.1', 8080
LEARNING_RATE = 3e-4
GAMMA = 0.99
EPS_CLIP = 0.2
K_EPOCHS = 4        # Quante volte ripassare sui dati raccolti
BATCH_SIZE = 64     # Mini-batch per l'update
ROLLOUT_LEN = 2048  # Ogni quanti frame fermarsi per allenare
INPUT_DIM = 20      # Dimensione del tuo vettore di stato (da definire)
ACTION_DIM = 12     # Numero di azioni discrete (es. 4 direzioni x 3 bottoni)

# --- RETE NEURALE (Actor-Critic) ---
class PPOActorCritic(nn.Module):
    def __init__(self):
        super(PPOActorCritic, self).__init__()
        # Actor: Decide l'azione
        self.actor = nn.Sequential(
            nn.Linear(INPUT_DIM, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, ACTION_DIM),
            nn.Softmax(dim=-1)
        )
        # Critic: Valuta quanto è buona la situazione
        self.critic = nn.Sequential(
            nn.Linear(INPUT_DIM, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

    def forward(self):
        raise NotImplementedError
    
    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0) 
        probs = self.actor(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), self.critic(state)

    def evaluate(self, state, action):
        probs = self.actor(state)
        dist = Categorical(probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        return action_logprobs, state_values, dist_entropy

# --- UTILS PER SELF PLAY ---
def mirror_state(state_vec):
    """
    Inverte lo stato per il Player 2 in modo che sembri il Player 1.
    Esempio: Se state è [p1_x, p2_x, p1_hp, p2_hp],
    il mirrored sarà [p2_x, p1_x, p2_hp, p1_hp] (e coordinate X invertite se necessario).
    """
    # TODO: Implementare la logica specifica del tuo vettore di stato
    # Per ora facciamo una copia fittizia
    new_state = np.copy(state_vec)
    # Esempio: new_state[0], new_state[1] = new_state[1], new_state[0]
    return new_state

def decode_action(action_idx):
    """ Mappa l'indice 0-11 in comandi per il server Go """
    # Esempio semplificato
    moves = ["", "U", "D", "L", "R", "DL", "DR"]
    btns = ["", "a", "b"]
    
    # Logica per convertire un indice singolo in (move, btn)
    # Esempio: i primi 7 sono solo movimento, i successivi combinano con 'a', ecc.
    m = moves[action_idx % len(moves)]
    b = btns[action_idx // len(moves) % len(btns)]
    return m, b

# --- MAIN LOOP ---
def main():
    # 1. Setup Connessione
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    s.connect((HOST, PORT))
    reader = s.makefile('r', encoding='utf-8')
    print("✅ Connesso per PPO Self-Play")

    # 2. Setup Modello
    policy = PPOActorCritic()
    optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)
    
    # Memory Buffer
    memory_states = []
    memory_actions = []
    memory_logprobs = []
    memory_rewards = []
    memory_is_terminals = []

    timestep = 0
    
    # Loop infinito
    while True:
        # A. RICEZIONE STATO (Go -> Python)
        line = reader.readline()
        if not line: break
        
        # Parsifica il JSON in vettore (Implementa tu questa parte!)
        # raw_data = json.loads(line)
        # state_p1 = parse_to_vector(raw_data) 
        state_p1 = np.zeros(INPUT_DIM) # Placeholder
        
        # B. PREPARAZIONE STATI (Mirroring)
        state_p2 = mirror_state(state_p1) # P2 vede il mondo invertito

        # C. INFERENZA (Action Selection)
        # La stessa policy decide per entrambi!
        with torch.no_grad():
            action_p1, logprob_p1, _ = policy.act(state_p1)
            action_p2, logprob_p2, _ = policy.act(state_p2) # P2 usa la stessa intelligenza
        
        # D. INVIO AZIONI (Python -> Go)
        m1, b1 = decode_action(action_p1)
        m2, b2 = decode_action(action_p2)
        
        # Nota: per P2, se l'azione è "Sinistra" (relativa), 
        # potrebbe dover essere inviata come "Destra" assoluta al server, 
        # a meno che il server non gestisca input relativi.
        
        msg = json.dumps({
            "p1_move": m1, "p1_btn": b1,
            "p2_move": m2, "p2_btn": b2, # P2 sta giocando contro P1
            "reset": False
        }) + "\n"
        s.sendall(msg.encode('utf-8'))

        # E. SALVATAGGIO DATI (Solo per P1 per ora - Single Agent Perspective)
        # In Self-Play puro, potresti salvare anche i dati di P2,
        # ma spesso si allena "P1 contro una copia di se stesso".
        memory_states.append(state_p1)
        memory_actions.append(action_p1)
        memory_logprobs.append(logprob_p1)
        
        # TODO: Calcolare Reward REALE basato sul delta HP o vittoria
        reward = 0.1 # Placeholder
        memory_rewards.append(reward)
        memory_is_terminals.append(False) # Gestire fine round

        timestep += 1

        # F. AGGIORNAMENTO (PPO UPDATE)
        if timestep % ROLLOUT_LEN == 0:
            print(f"🔄 Training Update at step {timestep}...")
            
            # --- PPO ALGORITHM QUI ---
            # 1. Converti liste in tensori
            # 2. Calcola vantaggi (Monte Carlo o GAE)
            # 3. Loop per K epoche:
            #    - Loss = -min(surr1, surr2) + 0.5*MSE(val, targ) - 0.01*Entropy
            #    - optimizer.step()
            
            # Reset memoria dopo update
            memory_states = []
            memory_actions = []
            memory_logprobs = []
            memory_rewards = []
            memory_is_terminals = []
            
            # In questo momento il gioco in Go aspetterà la risposta del socket,
            # quindi il training è "bloccante" (pausa il gioco), che è perfetto.

if __name__ == "__main__":
    main()