import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml
from pathlib import Path
import json
import sys
import copy
from collections import deque
import struct

# Include parent folder to get env
sys.path.append(str(Path(__file__).resolve().parent.parent))

#Include now env
from pycode.environment import IkemenEnvironment

def vectorize(params):
    return torch.cat([p.reshape(-1) for p in params])

configsPath = Path(__file__).resolve().parent.parent / 'configs.yaml'

with open(configsPath, 'r') as configsFile:
    CONFIGS = yaml.safe_load(configsFile)

class TeacherModel(nn.Module):
    def __init__(self, env, configs=CONFIGS, device='cpu'):
        super().__init__()
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
            deviceName = torch.cuda.get_device_name(torch.cuda.current_device())
            print(f"Using device: {deviceName}")
        else:
            self.device = torch.device('cpu')
            print(f"Using device: cpu")
        
        self.configs = configs
        
        self.gamma = configs['stdPPO']['gamma']
        self.gae_lambda = configs['stdPPO']['gae_lambda']
        self.episodes = 100
        self.lr = configs['stdPPO']['lr']
        self.clip_epsilon = configs['stdPPO']['clip_epsilon']
        self.entropy_coef = configs['stdPPO']['entropy_coef']
        self.value_loss_coef = configs['stdPPO']['value_loss_coef']
        self.max_grad_norm = configs['stdPPO']['max_grad_norm']
        self.update_epochs = configs['TeachTrain']['update_epochs']
        self.batch_size = configs['TeachTrain']['batch_size']
        self.minibatch_size = configs['TeachTrain']['minibatch_size']
        self.rollout_steps = configs['TeachTrain']['rollout_steps']
        
        state_dim = env.observation_space.shape[0]
        self.actionsMove = env.action_space.nvec[0]
        self.actionsHit = env.action_space.nvec[1]
        
        # Not using the predefined model in stateNetwork.py. Creating a new model here (and then modify the module later on)
        self.featureExtractor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh()
        )
        
        self.movePolicy = nn.Linear(128, self.actionsMove)
        self.hitPolicy = nn.Linear(128, self.actionsHit)
        self.valueEstimator = nn.Linear(128, 1)
        self.to(self.device)
        
    # Changing momentarily any additional functions. Recover them in previous commits if needed.
    
    def compute_gae(self, rewards, values, dones, next_value):
        """Calcola i vantaggi usando GAE (Generalized Advantage Estimation)."""
        advantages = torch.zeros_like(rewards)
        lastgaelam = 0
        
        # Si itera all'indietro
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                nextnonterminal = 1.0 - dones[t] # Se done=1, next è 0
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t+1]
                nextvalues = values[t+1]
                
            delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            
        returns = advantages + values
        return advantages, returns

    def ppo_update(self, model, optimizer, batch_data):
        """Esegue l'aggiornamento dei pesi della rete."""
        # Scompattiamo i dati del buffer
        b_obs, b_actions, b_logprobs, b_returns, b_advantages, b_values = batch_data

        current_batch_size = b_obs.shape[0]
        
        # Ciclo di epoche (Solitamente 4 o 10)
        for epoch in range(self.update_epochs):
            # Generiamo indici casuali per i mini-batch
            indices = np.random.permutation(current_batch_size)
            
            for start in range(0, current_batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_idxs = indices[start:end]

                # 1. Forward pass sui dati salvati (Ricalcoliamo logprob e values attuali)
                # Nota: model.get_action_and_value deve restituire (logits, value)
                _, new_logprobs, entropy, new_values = model.get_action_and_value(b_obs[mb_idxs], b_actions[mb_idxs])
                
                # 2. Calcolo Ratio (Probabilità Nuova / Probabilità Vecchia)
                logratio = new_logprobs - b_logprobs[mb_idxs]
                ratio = logratio.exp()

                # 3. Calcolo PPO Loss (Clipped)
                mb_advantages = b_advantages[mb_idxs]
                # Normalizzazione vantaggi (stabilizza il training)
                if mb_advantages.std() > 1e-8:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                # 4. Value Loss (MSE tra valore predetto e rendimento reale)
                # Spesso si usa anche il clipping sulla value function, qui metto la versione base
                value_loss = 0.5 * ((new_values - b_returns[mb_idxs]) ** 2).mean()

                # 5. Totale
                loss = policy_loss - (self.entropy_coef * entropy.mean()) + (self.value_loss_coef * value_loss)

                # 6. Backpropagation
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                optimizer.step()
    
    def get_action_and_value(self, x, action=None):
        """Metodo fondamentale per PPO: restituisce azioni, log_prob e value."""
        hidden = self.featureExtractor(x)
        
        # 1. Calcolo Logits per le due teste
        logits_move = self.movePolicy(hidden)
        logits_attack = self.hitPolicy(hidden)
        
        # 2. Creazione Distribuzioni
        dist_move = torch.distributions.Categorical(logits=logits_move)
        dist_attack = torch.distributions.Categorical(logits=logits_attack)
        
        if action is None:
            # Inferenza: Campioniamo le azioni
            action_move = dist_move.sample()
            action_attack = dist_attack.sample()
            action = torch.stack([action_move, action_attack], dim=1)
        else:
            # Training: Usiamo le azioni passate
            action_move = action[:, 0]
            action_attack = action[:, 1]
            
        # 3. Calcolo Log Probabilità (Somma dei logaritmi per eventi indipendenti)
        log_prob = dist_move.log_prob(action_move) + dist_attack.log_prob(action_attack)
        
        # 4. Entropia
        entropy = dist_move.entropy() + dist_attack.entropy()
        
        # 5. Valore
        value = self.valueEstimator(hidden)
        
        return action, log_prob, entropy, value 
    
    def setParameters(self, parametersVector):
        index = 0
        with torch.no_grad():
            for param in self.parameters():
                n = param.numel()
                param.copy_(parametersVector[index:index + n].view(param.size()))
                index += n

    def save(self):
        torch.save(self.state_dict(), 'teacher_model.pt')

    def load(self):
        self.load_state_dict(torch.load('teacher_model.pt', map_location=self.device))

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret
    
    def act(self, state, mode='exploit'):
        networkInput = state
        with torch.no_grad():
            logits_move = self.movePolicy(networkInput)
            logits_attack = self.hitPolicy(networkInput)
        if mode == 'explore':
            dist_move = torch.distributions.Categorical(logits=logits_move)
            dist_attack = torch.distributions.Categorical(logits=logits_attack)
            action_move = dist_move.sample()
            action_attack = dist_attack.sample()
            action = torch.stack([action_move, action_attack], dim=1)
        else:
            action_move = logits_move.argmax(dim=-1)
            action_attack = logits_attack.argmax(dim=-1)
            action = torch.stack([action_move, action_attack], dim=1)

        return action.item()

    # TODO : Clean accordingly...
    def runEpisode(self, env, last_obs, rollout_steps, opponent_model):
        """
        Raccoglie dati simulando un loop 'step' manuale poiché l'env non ne ha uno unificato.
        """
        b_obs, b_actions, b_logprobs, b_rewards, b_dones, b_values = [], [], [], [], [], []
        
        batch_wins = 0
        batch_matches = 0

        # Assicuriamoci che last_obs sia un tensore sulla GPU/CPU corretta
        obs_tensor = torch.tensor(last_obs, dtype=torch.float32).to(self.device).unsqueeze(0)

        # Assicuriamoci che l'env abbia uno stato precedente per il calcolo del reward iniziale
        if env.previousState is None:
             # Hack: se non c'è previousState, usiamo quello che abbiamo appena normalizzato (non perfetto ma evita crash)
             pass 

        with torch.no_grad():
            for step in range(rollout_steps):
                # --- 1. POLICY INFERENZA ---
                # Player 1 (Teacher)
                a1, logp1, _, v1 = self.get_action_and_value(obs_tensor)
                
                # Player 2 (Opponent) - Vede lo stato FLIPPATO
                opp_obs_tensor = self.flip_observation(obs_tensor)
                a2, _, _, _ = opponent_model.get_action_and_value(opp_obs_tensor)
                
                # --- 2. PREPARAZIONE AZIONI ---
                # Il modello restituisce tensori [MoveIdx, BtnIdx]. L'env vuole tuple (int, int).
                # Convertiamo: Tensor GPU -> Numpy -> Lista -> Tupla
                act_p1_tuple = tuple(a1.cpu().numpy().flatten().tolist()) # Es: (1, 0)
                act_p2_tuple = tuple(a2.cpu().numpy().flatten().tolist()) 

                # --- 3. INTERAZIONE ENV (Manuale) ---
                # A. Invia azioni
                env.executeAction(act_p1_tuple, act_p2_tuple)
                
                # B. Ricevi nuovo stato grezzo
                # Nota: needFrame=False perché il teacher usa solo vettori
                try:
                    raw_next_state, _ = env.recieve(needFrame=False)
                except (ConnectionError, struct.error) as e:
                    print(f"Errore ricezione dati: {e}. Interrompo rollout.")
                    break

                # C. Calcola Reward e Done
                # Nota: rewardCompute usa env.previousState. Dobbiamo assicurarci che sia settato.
                # Se env.recieve non aggiorna env.previousState, lo facciamo noi alla fine del ciclo.
                reward, done = env.rewardCompute(raw_next_state)
                
                # D. Normalizza il nuovo stato per la rete neurale
                next_obs_numpy = env.normalizeState(raw_next_state)
                
                # --- 4. SALVATAGGIO DATI ---
                b_obs.append(obs_tensor)
                b_actions.append(a1)
                b_logprobs.append(logp1)
                b_values.append(v1.flatten())
                b_rewards.append(torch.tensor(reward, dtype=torch.float32).to(self.device))
                b_dones.append(torch.tensor(done, dtype=torch.float32).to(self.device))
                
                # --- 5. GESTIONE FINE EPISODIO (RESET) ---
                if done:
                    # Tracking
                    batch_matches += 1
                    if reward > 0: batch_wins += 1 # Assumendo reward positivo per vittoria
                    
                    # Reset dell'ambiente
                    env.reset()
                    
                    # Dopo il reset, dobbiamo ricevere il nuovo stato iniziale pulito
                    try:
                        raw_reset_state, _ = env.recieve(needFrame=False)
                        next_obs_numpy = env.normalizeState(raw_reset_state)
                        # Resettiamo anche il previousState dell'env per evitare reward enormi al primo frame
                        env.previousState = raw_reset_state 
                    except Exception as e:
                        print(f"Errore durante reset: {e}")
                        break
                else:
                    # Se non è done, aggiorniamo il previousState per il prossimo calcolo reward
                    env.previousState = raw_next_state

                # Aggiorniamo il tensore corrente per il prossimo step
                obs_tensor = torch.tensor(next_obs_numpy, dtype=torch.float32).to(self.device).unsqueeze(0)

            # --- 6. BOOTSTRAPPING (Valore finale) ---
            _, _, _, next_value = self.get_action_and_value(obs_tensor)
            next_value = next_value.flatten()

        # --- 7. IMPACCHETTAMENTO ---
        t_obs = torch.stack(b_obs)
        t_actions = torch.stack(b_actions)
        t_logprobs = torch.stack(b_logprobs)
        t_values = torch.stack(b_values)
        t_rewards = torch.stack(b_rewards)
        t_dones = torch.stack(b_dones)

        advantages, returns = self.compute_gae(t_rewards, t_values, t_dones, next_value)

        flat_obs = t_obs.view(-1, t_obs.shape[-1])
        flat_actions = t_actions.view(-1, t_actions.shape[-1])
        flat_logprobs = t_logprobs.view(-1)
        flat_returns = returns.view(-1)
        flat_advantages = advantages.view(-1)
        flat_values = t_values.view(-1)

        batch_data = (flat_obs, flat_actions, flat_logprobs, flat_returns, flat_advantages, flat_values)
        
        # Calcolo Win Rate
        win_rate = batch_wins / batch_matches if batch_matches > 0 else 0.0
        
        # Restituiamo next_obs (in formato numpy) per mantenere la continuità nel loop principale
        return batch_data, next_obs_numpy, win_rate

    def flip_observation(self, obs):
        """ Flips observation for opponent's perspective """
        
        flipped = obs.clone()

        # From our 12-dim observation vec:
        # 0: P1 HP,       1: P2 HP
        # 2: Rel X,       3: Rel Y
        # 4: P1 Abs X,    5: P2 Abs X
        # 6: P1 Facing,   7: P2 Facing
        # 8: P1 Power,    9: P2 Power
        # 10: P1 Anim,    11: P2 Anim

        # HP
        flipped[..., 0] = obs[..., 1]
        flipped[..., 1] = obs[..., 0]
        
        # Absolute X
        flipped[..., 4] = obs[..., 5]
        flipped[..., 5] = obs[..., 4]
        
        # Facing
        flipped[..., 6] = obs[..., 7]
        flipped[..., 7] = obs[..., 6]
        
        # Power
        flipped[..., 8] = obs[..., 9]
        flipped[..., 9] = obs[..., 8]
        
        # Anim
        flipped[..., 10] = obs[..., 11]
        flipped[..., 11] = obs[..., 10]

        # X distance flip: Se P2 è a dx (+), per P2 P1 è a sx (-)
        flipped[..., 2] = -obs[..., 2]
        
        # Y distance flip: Se P2 è sopra (+), per P2 P1 è sotto (-)
        flipped[..., 3] = -obs[..., 3]

        return flipped    
    
    def trainPPO(self):
        print(f"Start Self-Play Training on {self.device}")
        
        # 1. SETUP ENV & CONNECTION
        # Inizializziamo l'ambiente
        env = IkemenEnvironment(training_mode='teacher')
        try:
            env.launch_game()
            env.connect()
            print("Environment connected successfully.")
        except ConnectionError as e:
            print(f"Critical Error: Could not connect to environment. {e}")
            env.close_game()
            return

        # 2. SETUP OPTIMIZER & OPPONENT
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        
        # Opponent (Copia congelata)
        opponent_model = copy.deepcopy(self)
        opponent_model.to(self.device)
        opponent_model.eval()
        for param in opponent_model.parameters():
            param.requires_grad = False

        # 3. PREPARAZIONE STATO INIZIALE
        # Facciamo un reset e una prima lettura per avere 'last_obs' valido
        env.reset()
        try:
            raw_init, _ = env.recieve(needFrame=False)
            last_obs = env.normalizeState(raw_init)
            env.previousState = raw_init # Inizializziamo per il reward
        except Exception as e:
            print(f"Error getting initial state: {e}")
            env.close_game()
            return

        # Variabili Loop
        total_updates = self.episodes
        global_step = 0
        win_rate_history = deque(maxlen=5)

        # --- TRAINING LOOP ---
        for update in range(1, total_updates + 1):
            
            # A. RACCOLTA DATI
            batch_data, next_obs, win_rate = self.runEpisode(
                env, 
                last_obs, 
                self.rollout_steps, 
                opponent_model
            )
            
            # Aggiorniamo stato e contatori
            last_obs = next_obs
            global_step += self.rollout_steps
            
            # Tracking
            win_rate_history.append(win_rate)
            avg_win_rate = sum(win_rate_history) / len(win_rate_history) if len(win_rate_history) > 0 else 0.0
            
            # B. UPDATE PPO (LEARNER)
            self.ppo_update(self, optimizer, batch_data)
            
            # C. LOGGING
            avg_return = batch_data[3].mean().item()
            print(f"Update {update}/{total_updates} | Steps: {global_step} | "
                  f"Avg Return: {avg_return:.3f} | Win Rate: {avg_win_rate:.2%}")

            # D. OPPONENT UPGRADE LOGIC
            # Se il learner vince > 60% delle volte, diventa il nuovo maestro
            if avg_win_rate > 0.60 and len(win_rate_history) == 5:
                print("🚀 UPGRADE: Opponent updated to current Learner policy.")
                opponent_model.load_state_dict(self.state_dict())
                win_rate_history.clear()
            
            # E. SAVE CHECKPOINT
            if update % 10 == 0:
                self.save()
                print(f"Checkpoint saved at update {update}")

        print("Training completed.")
        env.close_game()