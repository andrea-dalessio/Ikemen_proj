from pathlib import Path
from collections import deque
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import builtins
from pandas import DataFrame
import csv


originalPrint = print

def vectorize(params):
    return torch.cat([p.reshape(-1) for p in params])

def safe_print(*args, **kwargs):
    tqdm.write(" ".join(map(str, args)), **kwargs)

class ResidualBlock(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.fc = nn.Linear(size, size)
        self.ln = nn.LayerNorm(size)
    
    def forward(self, x):
        # Skip connection: f(x) + x
        return F.relu(self.ln(self.fc(x))) + x

class TeacherModel(nn.Module):
    def __init__(self, env, configs, load_checkpoint=False):
        super().__init__()
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
            deviceName = torch.cuda.get_device_name(torch.cuda.current_device())
            print(f"Using device: {deviceName}")
        else:
            self.device = torch.device('cpu')
            print(f"Using device: cpu")
        
        self.configs = configs
        self.namespace = configs['teacherModel']['namespace']
        self.savedir = f'{os.getcwd()}/models_saves_t'
        self.loggingPath = f'{os.getcwd()}/logs/{self.namespace}_training_logs.csv'
        
        self.gamma = configs['general']['gamma']
        self.gae_lambda = configs['general']['gae_lambda']
        self.episodes = configs['teacherModel']['episodes']
        self.lr = configs['general']['lr']
        self.clip_epsilon = configs['general']['clip_epsilon']
        self.entropy_coef = configs['general']['entropy_coef']
        self.value_loss_coef = configs['general']['value_loss_coef']
        self.max_grad_norm = configs['general']['max_grad_norm']
        self.update_epochs = configs['teacherModel']['update_epochs']
        self.batch_size = configs['teacherModel']['batch_size']
        self.minibatch_size = configs['teacherModel']['minibatch_size']
        self.rollout_steps = configs['teacherModel']['rollout_steps']
        self.checkpoint = 0
        
        self.state_dim = env.state_space[1]
        self.actionsMove = env.action_space[1]
        self.actionsHit = env.action_space[2]
        
        self.env = env
        if self.env.count is not None:
            self.env_number = self.env.count
        else:
            self.env_number = 1 
        
        self.input_layer = nn.Linear(self.state_dim, 512)
        
        # Residual Blocks
        self.res_block1 = ResidualBlock(512)
        self.res_block2 = ResidualBlock(512)
        self.res_block3 = ResidualBlock(512)
        
        self.feature_head = nn.Linear(512, 256)
        
        # Policy Heads (actor-critic)
        self.movePolicy = nn.Linear(256, self.actionsMove)
        self.hitPolicy = nn.Linear(256, self.actionsHit)
        self.valueEstimator = nn.Linear(256, 1)

        self.to(self.device)
        
        if load_checkpoint and len(os.listdir(Path(self.savedir)))>0:
            self.load()
            print("Loaded Teacher Model from checkpoint.")
        else:
            print("Initialized new Teacher Model.")
        
    def make_copy(self):
        model = TeacherModel(self.env, self.configs)
        model.input_layer.load_state_dict(self.input_layer.state_dict())
        model.res_block1.load_state_dict(self.res_block1.state_dict())
        model.res_block2.load_state_dict(self.res_block2.state_dict())
        model.res_block3.load_state_dict(self.res_block3.state_dict())
        model.feature_head.load_state_dict(self.feature_head.state_dict())
        model.hitPolicy.load_state_dict(self.hitPolicy.state_dict())
        model.movePolicy.load_state_dict(self.movePolicy.state_dict())
        return model
    
    # Changing momentarily any additional functions. Recover them in previous commits if needed.
    def compute_gae(self, rewards, values, dones, next_value):
        T, N = rewards.shape
        advantages = torch.zeros_like(rewards)
        lastgaelam = torch.zeros(N, device=self.device)

        # iterate backwards in time
        for t in reversed(range(T)):
            if t == T - 1:
                nextnonterminal = 1.0 - dones[t]
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]

            delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
            lastgaelam = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            advantages[t] = lastgaelam

        returns = advantages + values
        return advantages, returns

    def ppo_update(self, model, optimizer, batch_data, last_state):
        """Esegue l'aggiornamento dei pesi della rete."""
        # Scompattiamo i dati del buffer
        b_state, b_actions, b_logprobs, b_returns, b_advantages, _ = batch_data

        current_batch_size = b_state.shape[0]
        
        last_ping_time = time.time()
        ping_interval = 2.0  # seconds
        
        # Ciclo di epoche
        builtins.print = safe_print
        for _ in tqdm(range(self.update_epochs), desc="Consuming batches"):
            indices = np.random.permutation(current_batch_size)
            
            for start in range(0, current_batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_idxs = indices[start:end]

                # --- CODICE PPO STANDARD ---
                _, new_logprobs, entropy, new_values = model.get_action_and_value(b_state[mb_idxs], b_actions[mb_idxs])
                logratio = new_logprobs - b_logprobs[mb_idxs]
                ratio = logratio.exp()
                
                mb_advantages = b_advantages[mb_idxs]
                
                if mb_advantages.std() > 1e-8:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                value_loss = 0.5 * ((new_values - b_returns[mb_idxs]) ** 2).mean()
                
                loss = policy_loss - (self.entropy_coef * entropy.mean()) + (self.value_loss_coef * value_loss)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                optimizer.step()
                # ---------------------------

                # --- KEEP ALIVE CHECK ---
                # Se è passato troppo tempo dall'ultimo contatto con il gioco
                if time.time() - last_ping_time > ping_interval:
                    last_state = self.keep_alive(last_state)
                    last_ping_time = time.time()
        
        
        builtins.print = originalPrint
        return last_state, value_loss.item() # Return updated last_obs after keep-alive
    
    def keep_alive(self, current_obs):
        """
        Needed to keep the connection alive with the Ikemen env while the training runs!
        """
        # Generating no-op actions to keep servers alive
        n_envs = self.env_number
        dummy_action = np.zeros((n_envs, 2), dtype=int) 
        
        try:
            self.env.executeAction(dummy_action, dummy_action)
            
            raw_next_states, _ = self.env.recieve()
            
            # Get new state to compute update
            if self.env.count > 1:
                # Case SuperEnvironment
                self.env.envs[0].previousState = raw_next_states[0]
                # Note: SuperEnv handles previousState for all envs internally.
            else:
                self.env.previousState = raw_next_states
                

            return self.env.normalizeState(raw_next_states)
            
        except Exception as e:
            print(f"Warning: Keep-alive failed: {e}")
            return current_obs
    
    def get_action_and_value(self, x, action=None):
        """Metodo fondamentale per PPO: restituisce azioni, log_prob e value."""
        x = F.relu(self.input_layer(x))
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        hidden = F.relu(self.feature_head(x))
        
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
        else:
            # Training: Usiamo le azioni passate
            action_move = action[:, 0]
            action_attack = action[:, 1]
        
        # print("[Master]> ", action_move, action_attack)
        
        new_action = torch.stack([action_move, action_attack], dim=1)
        # 3. Calcolo Log Probabilità (Somma dei logaritmi per eventi indipendenti)
        log_prob = dist_move.log_prob(action_move) + dist_attack.log_prob(action_attack)
        
        # 4. Entropia
        entropy = dist_move.entropy() + dist_attack.entropy()
        
        # 5. Valore
        value = self.valueEstimator(hidden)
        
        return new_action, log_prob, entropy, value 
    
    def setParameters(self, parametersVector):
        index = 0
        with torch.no_grad():
            for param in self.parameters():
                n = param.numel()
                param.copy_(parametersVector[index:index + n].view(param.size()))
                index += n

    def save(self, id):
        torch.save(self.state_dict(), f"{self.savedir}/{self.namespace}_{id}.pt")

    def load(self):
        saves = os.listdir(self.savedir)
        chosen = saves[-1]
        end = chosen.split("_")[-1]
        id = end.split(".")[0]
        if id.isdigit():
            self.checkpoint = int(id)
        print(f"[Teacher]> {self.savedir}/{chosen}")
        self.load_state_dict(torch.load(f"{self.savedir}/{chosen}", map_location=self.device))

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret
    
    def act(self, state, mode='exploit'):
        networkInput = state
        with torch.no_grad():
            x = F.relu(self.input_layer(networkInput))
            x = self.res_block1(x)
            x = self.res_block2(x)
            x = self.res_block3(x)
            hidden = F.relu(self.feature_head(x))
            logits_move = self.movePolicy(hidden)
            logits_attack = self.hitPolicy(hidden)
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
    def runEpisode(self, last_state, rollout_steps, opponent_model):
        """
        Raccoglie dati simulando un loop 'step' manuale poiché l'env non ne ha uno unificato.
        """
        b_states = [] 
        b_actions = [] 
        b_logprobs = []
        b_rewards = [] 
        b_dones = [] 
        b_values = []
        
        batch_wins = 0
        batch_matches = 0

        crash_occurred = False
        
        # Assicuriamoci che last_obs sia un tensore sulla GPU/CPU corretta
        state_tensor = torch.tensor(last_state, dtype=torch.float32).to(self.device)
        if self.env_number == 1:
            state_tensor = state_tensor.unsqueeze(0)

        # Assicuriamoci che l'env abbia uno stato precedente per il calcolo del reward iniziale
        if not self.env.hasPreviousState:
            self.env.setPreviousState(last_state.copy())

        with torch.no_grad():
            for i in tqdm(range(rollout_steps), desc="Episode rollout progess"):
                # --- 1. POLICY INFERENZA ---
                # Player 1 (Teacher)
                a1, logp1, _, v1 = self.get_action_and_value(state_tensor)
                
                # Player 2 (Opponent) - Vede lo stato FLIPPATO
                opp_obs_tensor = self.flip_observation(state_tensor)
                a2, _, _, _ = opponent_model.get_action_and_value(opp_obs_tensor)
                
                # --- 2. PREPARAZIONE AZIONI ---
                act_p1 = a1.cpu().numpy()
                act_p2 = a2.cpu().numpy()

                # --- 3. INTERAZIONE ENV (Manuale) ---
                # A. Invia azioni
                try:
                
                    self.env.executeAction(act_p1, act_p2)
                except Exception as e:
                    print(f"Errore invio azioni: {e}. Interrompo rollout.")
                    crash_occurred = True
                    break
                try:
                    raw_next_state, _ = self.env.recieve()
                except Exception as e:
                    print(f"Errore recezione stato: {e}. Interrompo rollout.")
                    crash_occurred = True
                    break

                # C. Calcola Reward e Done
                # Nota: rewardCompute usa env.previousState. Dobbiamo assicurarci che sia settato.
                # Se env.recieve non aggiorna env.previousState, lo facciamo noi alla fine del ciclo.
                rewards, dones = self.env.rewardCompute(raw_next_state)
                rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
                dones = torch.tensor(dones, dtype=torch.float32, device=self.device)      
                
                # --- 4. SALVATAGGIO DATI ---
                b_states.append(state_tensor)
                b_actions.append(a1)
                b_logprobs.append(logp1)
                b_values.append(v1.squeeze(-1))
                b_rewards.append(rewards.float())
                b_dones.append(dones.float())
                
                
                next_state = self.env.normalizeState(raw_next_state)
                state_tensor = torch.tensor(next_state, dtype=torch.float32).to(self.device)
                
                # --- 5. GESTIONE FINE EPISODIO (RESET) ---
                for i, done in enumerate(dones):
                    if done:
                        batch_matches += 1
                        env_state = raw_next_state[i]
                        if env_state is not None:
                            p1_hp = env_state.get('p1_hp', 0)
                            p2_hp = env_state.get('p2_hp', 0)
                            if p1_hp > p2_hp: 
                                batch_wins += 1
                        try:
                            raw_reset_state, _ = self.env.reset(i)
                        except Exception as e:
                            print(f"Errore durante reset: {e}")
                            break
                        normedState = self.env.normalizeState(raw_reset_state, i)
                        state_tensor[i] = torch.tensor(normedState, dtype=torch.float32, device=self.device)
                    else:
                        # Se non è done, aggiorniamo il previousState per il prossimo calcolo reward
                        self.env.setPreviousState(raw_next_state)

                # Aggiorniamo il tensore corrente per il prossimo step
            
            
            # --- 6. BOOTSTRAPPING (Valore finale) ---
            _, _, _, next_value = self.get_action_and_value(state_tensor)
            next_value = next_value.flatten()

        if crash_occurred or len(b_states) == 0:
            print("WARNING: No data collected in rollout. Skipping update.")
            return None,  None, 0.0, True

        # --- 7. IMPACCHETTAMENTO ---
        t_states = torch.stack(b_states)
        t_actions = torch.stack(b_actions)
        t_logprobs = torch.stack(b_logprobs)
        t_values = torch.stack(b_values)
        t_rewards = torch.stack(b_rewards)
        t_dones = torch.stack(b_dones)

        advantages, returns = self.compute_gae(t_rewards, t_values, t_dones, next_value)
        
        
        # ----- Flattening ------
        T, N, _ = t_states.shape
        flat_states = t_states.view(T * N, self.state_dim)
        flat_actions = t_actions.view(T * N, -1)
        flat_logprobs = t_logprobs.view(T * N)
        flat_values = t_values.view(T * N)
        flat_returns = returns.view(T * N)
        flat_advantages = advantages.view(T * N)

        batch_data = (
            flat_states, 
            flat_actions, 
            flat_logprobs, 
            flat_returns, 
            flat_advantages, 
            flat_values
        )
        
        # Calculate Win Rate
        if batch_matches > 0:
            win_rate = batch_wins / batch_matches
        else:
            win_rate = None
        
        # Return next_obs (in numpy format) to maintain continuity in the main loop
        return batch_data, next_state, win_rate, False

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
        flipped[..., 4] = 1.0 - obs[..., 5]
        flipped[..., 5] = 1.0 - obs[..., 4]
        
        # Facing
        flipped[..., 6] = -obs[..., 7]
        flipped[..., 7] = -obs[..., 6]
        
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

        # Time (13th extra dimension) remains unchanged but still copy it
        flipped[..., 12] = obs[..., 12]
        
        # Added just now: speeds! The model isn't an RNN so to "Markovize" the state we add speeds as input features
        flipped[..., 13] = -obs[..., 15]  # p2_dx flipped to p1_dx
        flipped[..., 14] = obs[..., 16]  # p2_dy flipped to p1_dy
        flipped[..., 15] = -obs[..., 13]  # p1_dx flipped to p2_dx
        flipped[..., 16] = obs[..., 14]  # p1_dy flipped to p2_dy
        
        flipped[..., 17] = obs[..., 18]  # p1_y
        flipped[..., 18] = obs[..., 17]  # p2_y
        
        return flipped    
    
    def trainPPO(self):
        print(f"Start Self-Play Training on {self.device}")
        opponent_model = self.make_copy()
        opponent_model.to(self.device)
        opponent_model.eval()
        for param in opponent_model.parameters():
            param.requires_grad = False
        
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        
        # 1. SETUP ENV & CONNECTION
        # Inizializziamo l'ambiente
        
        try:
            self.env.start()
            
            raw_init, _ = self.env.wait_for_match_start()
            self.env.setPreviousState(raw_init) # Inizializziamo per il reward
            state = self.env.normalizeState(raw_init)
            
            print("Environment connected successfully.")
        except Exception as e:
            print(f"Critical Error: Could not connect to environment. {repr(e)}")
            self.env.close_game()
            return

        # 2. SETUP OPTIMIZER & OPPONENT
        
        
        # Opponent (Copia congelata)



        # Variabili Loop
        total_updates = self.episodes
        global_step = 0
        win_rate_history = deque(maxlen=5)

        # --- TRAINING LOOP ---
        print("Start episode loop")
        for update in range(self.checkpoint, total_updates - self.checkpoint):
            
            # A. RACCOLTA DATI
            
            print(f"[Master]> Update {update + 1}: start episode")
            
            try:
                builtins.print = safe_print
                batch_data, next_state, win_rate, crash_occurred = self.runEpisode(state, self.configs['teacherModel']['rollout_steps'], opponent_model)
            finally:
                builtins.print = originalPrint
            
            if crash_occurred or batch_data is None:
                print(f"[Master]> Update {update+1}: Skipping update due to {'crash' if crash_occurred else 'lack of data'}. Resyncing...")
                try:
                    raw_new, _ = self.env.hard_restart()
                    state = self.env.normalizeState(raw_new)
                    self.env.setPreviousState(raw_new)
                    print("Resync successful. Continuing training.")
                except Exception as e:
                    print(f"Critical Error during resync: {e}. Ending training.")
                    break
                continue
            

            
            # Aggiorniamo stato e contatori
            state = next_state
            global_step += self.rollout_steps
            
            # Tracking
            if win_rate is not None:
                win_rate_history.append(win_rate)
                
            if len(win_rate_history) > 0:
                avg_win_rate = sum(win_rate_history) / len(win_rate_history)
            else:
                avg_win_rate = 0.0
            
            # B. UPDATE PPO (LEARNER)
            print(f"[Master]> Coumputing PPO update")
            state, valueloss = self.ppo_update(self, optimizer, batch_data, state)
            
            # C. LOGGING
            if batch_data is not None:
                avg_return = batch_data[3].mean().item()
            else:
                print("[Master]> Batch empty: skip")
                continue
            print(f"[Master]> Update {update + 1}/{total_updates} | Steps: {global_step} | Avg Return: {avg_return:.3f} | Win Rate: {avg_win_rate:.2%}")

            # D. OPPONENT UPGRADE LOGIC
            # Se il learner vince > 60% delle volte, diventa il nuovo maestro
            if avg_win_rate > 0.60 and len(win_rate_history) == 5:
                print("🚀 UPGRADE: Opponent updated to current Learner policy.")
                opponent_model.load_state_dict(self.state_dict())
                win_rate_history.clear()
            
            # E. SAVE CHECKPOINT
            if (update + 1) % 10 == 0:
                self.save(update + 1)
                print(f"Checkpoint saved at update {update + 1}")
                
            # F. LOG TO CSV
            log_data = {
                'update': update + 1,
                'steps': global_step,
                'avg_return': avg_return,
                'win_rate': avg_win_rate,
                'value_loss': valueloss
            }
            df = DataFrame([log_data])
            df.to_csv(self.loggingPath, mode='a', header=not Path(self.loggingPath).is_file(), index=False)

        print("[Master]> Training completed.")
        self.env.close_game()