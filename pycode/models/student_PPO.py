from pathlib import Path
from collections import deque
import numpy as np
import os
import time
import torch
import torch.nn as nn
import builtins
from tqdm import tqdm
import torchvision.transforms as T
from pandas import DataFrame
from .networks import VisualDecisor
from .teacher_PPO import TeacherModel

def vectorize(params):
    return torch.cat([p.reshape(-1) for p in params])

def flip_frames(frames):
    if frames.dim() == 3:
        return torch.flip(frames, dims=[2])
    elif frames.dim() == 4:
        return torch.flip(frames, dims=[3])
    else:
        raise ValueError("Invalid frame shape")

def process_frame(frame_np):
    transform = T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    x = torch.from_numpy(frame_np).float()
    x = x/255
    if x.dim() == 3:
        if x.shape[2] == 4:
            x = x[:,:,:3]
        x = x.permute(2, 0, 1).unsqueeze(0)
    else:
        if x.shape[3] == 4:
            x = x[:,:,:,:3]
        x = x.permute(0, 3, 1, 2)

    return transform(x)

class StudentModel(nn.Module):
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
        self.namespace = configs['studentModel']['namespace']
        self.saveName = f'{os.getcwd()}/models_saves/{self.namespace}.pt'
        self.loggingPath = f'{os.getcwd()}/logs/{self.namespace}_training_logs.csv'
        
        self.gamma = configs['general']['gamma']
        self.gae_lambda = configs['general']['gae_lambda']
        self.episodes = configs['studentModel']['episodes']
        self.lr = configs['general']['lr']
        self.clip_epsilon = configs['general']['clip_epsilon']
        self.entropy_coef = configs['general']['entropy_coef']
        self.value_loss_coef = configs['general']['value_loss_coef']
        self.max_grad_norm = configs['general']['max_grad_norm']
        self.update_epochs = configs['studentModel']['update_epochs']
        self.batch_size = configs['studentModel']['batch_size']
        self.minibatch_size = configs['studentModel']['minibatch_size']
        self.rollout_steps = configs['studentModel']['rollout_steps']
        self.dist_temperature = configs['studentModel']['temperature']
        self.dist_coef = configs['studentModel']['dist_coef']
        
        self.frame_dim = env.observation_space
        self.actionsMove = env.action_space[1]
        self.actionsHit = env.action_space[2]
        
        print("Observation space: ",env.observation_space)
        print("Action space: ",env.action_space)
        
        self.env = env
        if self.env.count is not None:
            self.env_number = self.env.count
        else:
            self.env_number = 1 
        
        self.network = VisualDecisor(self.actionsMove, self.actionsHit)
        self.teacher = TeacherModel(self.env, self.configs, load_checkpoint=True)
        self.to(self.device)
        
        if load_checkpoint and Path(self.saveName).is_file():
            self.load()
            print("Loaded Student Model from checkpoint.")
        else:
            print("Initialized new Student Model.")
        
    def make_copy(self):
        model = StudentModel(self.env, self.configs)
        model.network.load_state_dict(self.network.state_dict())
        return model
    

        
    def act(self, frame, mode='exploit'):
        with torch.no_grad():
            logits_move, logits_attack = self.network.getMoveAndAttack(frame)
        if mode == 'explore':
            dist_move = torch.distributions.Categorical(logits=logits_move)
            dist_attack = torch.distributions.Categorical(logits=logits_attack)
            action_move = dist_move.sample()
            action_attack = dist_attack.sample()
        else:
            action_move = logits_move.argmax(dim=-1)
            action_attack = logits_attack.argmax(dim=-1)
        action = torch.stack([action_move, action_attack], dim=1)

        return action.item()
    
    def get_action_and_value(self, x, action=None):
        """Metodo fondamentale per PPO: restituisce azioni, log_prob e value."""
        logits_move, logits_attack, value = self.network.getMoveAndAttackAndValue(x)
        
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

        new_action = torch.stack([action_move, action_attack], dim=1)
            
        # 3. Calcolo Log Probabilità (Somma dei logaritmi per eventi indipendenti)
        log_prob = dist_move.log_prob(action_move) + dist_attack.log_prob(action_attack)
        
        # 4. Entropia
        entropy = dist_move.entropy() + dist_attack.entropy()
        
        return new_action, log_prob, entropy, value 

    def setParameters(self, parametersVector):
        index = 0
        with torch.no_grad():
            for param in self.parameters():
                n = param.numel()
                param.copy_(parametersVector[index:index + n].view(param.size()))
                index += n

    def save(self):
        torch.save(self.state_dict(), self.saveName)

    def load(self):
        self.load_state_dict(torch.load(self.saveName, map_location=self.device))

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret
    
    # Changing momentarily any additional functions. Recover them in previous commits if needed.
    def compute_gae(self, rewards, values, dones, next_value):
        T, N = rewards.shape
        advantages = torch.zeros_like(rewards)
        lastgaelam = torch.zeros(N)

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

    def ppo_update(self, optimizer, batch_data, last_state, last_frame):
        """Esegue l'aggiornamento dei pesi della rete."""
        # Scompattiamo i dati del buffer
        temp = self.dist_temperature
        b_frames, b_state, b_actions, b_logprobs, b_returns, b_advantages, _ = batch_data
        
        current_batch_size = b_state.shape[0]
        
        last_ping_time = time.time()
        ping_interval = 2.0  # seconds
        
        # Ciclo di epoche
        for _ in range(self.update_epochs):
            indices = np.random.permutation(current_batch_size)
            
            for start in range(0, current_batch_size, self.minibatch_size):
                torch.cuda.empty_cache()
                end = start + self.minibatch_size
                mb_idxs = indices[start:end]

                # --- CODICE PPO STANDARD ---
                actions_batch = b_actions[mb_idxs].to(self.device)
                frames_batch = b_frames[mb_idxs].to(self.device) #TODO
                _, s_logp, entropy, new_values = self.get_action_and_value(frames_batch, actions_batch)
                del frames_batch
                with torch.no_grad():
                    state_batch = b_state[mb_idxs].to(self.device)
                    _, t_logp, _, _ = self.teacher.get_action_and_value(state_batch, actions_batch)
                    del state_batch, actions_batch
                    
                entropy = entropy.cpu()
                new_values = new_values.cpu()
                s_logp = s_logp.cpu()
                t_logp = t_logp.cpu()
                    
                logratio = s_logp - b_logprobs[mb_idxs]
                ratio = logratio.exp()
                mb_advantages = b_advantages[mb_idxs]
                if mb_advantages.std() > 1e-8:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()
                value_loss = 0.5 * ((new_values - b_returns[mb_idxs]) ** 2).mean()
                ppo_loss = policy_loss - (self.entropy_coef * entropy.mean()) + (self.value_loss_coef * value_loss)
                distilation_loss = torch.sum(torch.exp(t_logp/temp) * (t_logp/temp - s_logp/temp),dim=-1)
                loss = ppo_loss + distilation_loss.mean()*(temp ** 2) * self.dist_coef
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                optimizer.step()
                # ---------------------------
                
        return last_frame, last_state, value_loss.item() # Return updated last_obs after keep-alive

    def runEpisode(self, state, frame, rollout_steps, opponent_model):
        
        b_frames = [] 
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
        frame_tensor = process_frame(frame) #TODO
        
        state_tensor = torch.tensor(state, dtype=torch.float32)
        if self.env_number == 1:
            state_tensor = state_tensor.unsqueeze(0)

        # Assicuriamoci che l'env abbia uno stato precedente per il calcolo del reward iniziale
        if not self.env.hasPreviousState:
            self.env.setPreviousState(state.copy())
        
        with torch.no_grad():
            for i in tqdm(range(rollout_steps), desc="Episode rollout progres"):
                
                frame_tensor_gpu = frame_tensor.to(self.device)
                a1, logp1, _, v1 = self.get_action_and_value(frame_tensor_gpu)
                
                opponent_screen = flip_frames(frame_tensor_gpu)
                a2, _, _, _ = opponent_model.get_action_and_value(opponent_screen)
                
                del frame_tensor_gpu, opponent_screen
                
                act_p1 = a1.cpu().numpy().astype(int)
                act_p2 = a2.cpu().numpy().astype(int)

                try:
                    self.env.executeAction(act_p1, act_p2)
                    
                except Exception as e:
                    print(f"Errore action execution: {e}")
                    crash_occurred = True
                    break

                try:
                    raw_next_state, next_frames = self.env.recieve()
                except Exception as e:
                    print(f"Errore reception: {e}. Interrompo rollout.")
                    crash_occurred = True
                    break
                # C. Calcola Reward e Done
                # Nota: rewardCompute usa env.previousState. Dobbiamo assicurarci che sia settato.
                # Se env.recieve non aggiorna env.previousState, lo facciamo noi alla fine del ciclo.
                rewards, dones = self.env.rewardCompute(raw_next_state)
                rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
                dones = torch.tensor(dones, dtype=torch.float32, device=self.device)      
                
                # --- 4. SALVATAGGIO DATI ---
                b_frames.append(frame_tensor) #TODO Check after uint shenaningans
                b_states.append(state_tensor)
                b_actions.append(a1)
                b_logprobs.append(logp1)
                b_values.append(v1.squeeze(-1))
                b_rewards.append(rewards.float())
                b_dones.append(dones.float())
                
                
                next_state = self.env.normalizeState(raw_next_state)
                state_tensor = torch.tensor(next_state, dtype=torch.float32)
                frame_tensor = process_frame(next_frames)  #TODO fix
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
                            raw_reset_state, reset_frames = self.env.reset(i)
                        except Exception as e:
                            print(f"Errore durante reset: {e}")
                            break
                        normedState = self.env.normalizeState(raw_reset_state, i)
                        state_tensor[i] = torch.tensor(normedState, dtype=torch.float32)
                        frame_tensor[i] = process_frame(reset_frames) #TODO fix
                    else:
                        # Se non è done, aggiorniamo il previousState per il prossimo calcolo reward
                        self.env.setPreviousState(raw_next_state)

                # Aggiorniamo il tensore corrente per il prossimo step
            # --- 6. BOOTSTRAPPING (Valore finale) ---
            frame_tensor_gpu = frame_tensor.to(self.device)
            _, _, _, next_value = self.get_action_and_value(frame_tensor_gpu)
            del frame_tensor_gpu
            
            next_value = next_value.flatten().cpu()

        if crash_occurred or len(b_frames) == 0:
            print("WARNING: No data collected in rollout. Skipping update.")
            return None, None, None, 0.0, True

        # --- 7. IMPACCHETTAMENTO ---
        t_frames = torch.stack(b_frames) #TODO Check after uint shenaningans
        t_states = torch.stack(b_states)
        t_actions = torch.stack(b_actions).cpu()
        t_logprobs = torch.stack(b_logprobs).cpu()
        t_values = torch.stack(b_values).cpu()
        t_rewards = torch.stack(b_rewards).cpu()
        t_dones = torch.stack(b_dones).cpu()

        advantages, returns = self.compute_gae(t_rewards, t_values, t_dones, next_value)
        
        
        # ----- Flattening ------
        T, N, C, H, W = t_frames.shape
        _, _, S = t_states.shape 
        flat_frames = t_frames.view(T * N, C, H, W)
        flat_states = t_states.view(T*N, S)
        flat_actions = t_actions.view(T * N, -1)
        flat_logprobs = t_logprobs.view(T * N)
        flat_values = t_values.view(T * N)
        flat_returns = returns.view(T * N)
        flat_advantages = advantages.view(T * N)

        batch_data = (
            flat_frames, 
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
        
        torch.cuda.empty_cache()
        return batch_data, next_frames, next_state, win_rate, False 
    
    def trainPPO(self):
        print(f"Start Self-Play Training on {self.device}")
        
        def safe_print(*args, **kwargs):
            tqdm.write(" ".join(map(str, args)), **kwargs)
        
        # Opponent (Copia congelata)
        opponent_model = self.make_copy()
        opponent_model.to(self.device)
        opponent_model.eval()
        for param in opponent_model.parameters():
            param.requires_grad = False
        
        # 2. SETUP OPTIMIZER & OPPONENT
        optimizer = torch.optim.Adam([
            {"params": self.network.backbone.layer4.parameters(), "lr": 1e-4},
            {"params": self.network.decisor.parameters(), "lr": 1e-3}
        ])

        # Variabili Loop
        total_updates = self.episodes
        global_step = 0
        win_rate_history = deque(maxlen=5)

        for update in range(1, total_updates + 1):
            torch.cuda.empty_cache()
            #os.system('clear')
            print(f"[Master]> Update {update}: start episode")
            # A. RACCOLTA DATI
            original_print = print
            
            try:
                self.env.start()
                
                raw_init, frame = self.env.wait_for_match_start()
                state = self.env.normalizeState(raw_init)
                self.env.setPreviousState(raw_init) # Inizializziamo per il reward
                
                print("Environment connected successfully.")
            except Exception as e:
                print(f"Critical Error: Could not connect to environment. {repr(e)}")
                self.env.close_game()
                return
            
            try:
                builtins.print = safe_print
                batch_data, next_frames, next_states, win_rate, _ = self.runEpisode(state, frame, self.configs['studentModel']['rollout_steps'], opponent_model)
            finally:
                builtins.print = original_print
            
            self.env.close_game()
            
            
            # Aggiorniamo stato e contatori
            last_frame = next_frames
            last_state = next_states
            global_step += self.rollout_steps
            
            # Tracking
            if win_rate is not None:
                win_rate_history.append(win_rate)
                
            if len(win_rate_history) > 0:
                avg_win_rate = sum(win_rate_history) / len(win_rate_history)
            else:
                avg_win_rate = 0.0
            
            # B. UPDATE PPO (LEARNER)
            valueloss = self.ppo_update(optimizer, batch_data, last_state, last_frame)
            
            # C. LOGGING
            if batch_data is not None:
                avg_return = batch_data[4].mean().item()
            print(f"[Master]> Update {update}/{total_updates} | Steps: {global_step} | Avg Return: {avg_return:.3f} | Win Rate: {avg_win_rate:.2%}")

            # D. OPPONENT UPGRADE LOGIC
            # Se il learner vince > 60% delle volte, diventa il nuovo maestro
            if avg_win_rate > 0.60 and len(win_rate_history) == 5:
                print(f"[Master]> Update {update}: Opponent updated to current Learner policy.")
                opponent_model.load_state_dict(self.state_dict())
                win_rate_history.clear()
            
            # E. SAVE CHECKPOINT
            if update % 10 == 0:
                self.save()
                print(f"[Master]> Update {update}: Checkpoint saved")
                
            # F. LOG TO CSV
            log_data = {
                'update': update,
                'steps': global_step,
                'avg_return': avg_return,
                'win_rate': avg_win_rate,
                'value_loss': valueloss
            }
            df = DataFrame([log_data])
            df.to_csv(self.loggingPath, mode='a', header=not Path(self.loggingPath).is_file(), index=False)

        print("[Master]> Training completed.")
        self.env.close_game()