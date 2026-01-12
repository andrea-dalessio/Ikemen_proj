import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml
from pathlib import Path
import json
from .stateNetwork import StateNetwork

def vectorize(params):
    return torch.cat([p.reshape(-1) for p in params])

configsPath = Path(__file__).resolve().parent / 'configs.yaml'

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
        self.episodes = ...
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

        # Ciclo di epoche (Solitamente 4 o 10)
        for epoch in range(self.update_epochs):
            # Generiamo indici casuali per i mini-batch
            indices = np.random.permutation(self.batch_size)
            
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_idxs = indices[start:end]

                # 1. Forward pass sui dati salvati (Ricalcoliamo logprob e values attuali)
                # Nota: model.get_action_and_value deve restituire (logits, value)
                _, new_logprobs, entropy, new_values = model.evaluate(b_obs[mb_idxs], b_actions[mb_idxs])
                
                # 2. Calcolo Ratio (Probabilità Nuova / Probabilità Vecchia)
                logratio = new_logprobs - b_logprobs[mb_idxs]
                ratio = logratio.exp()

                # 3. Calcolo PPO Loss (Clipped)
                mb_advantages = b_advantages[mb_idxs]
                # Normalizzazione vantaggi (stabilizza il training)
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
        hidden = self.feature_extractor(x)
        
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

    # TODO adapt the episode logic to ikemen
    def runEpisode(self, env):
        # states   = []
        # actions  = []
        # rewards  = []
        # logProbabilities = []
        
        # state, _ = self.env.reset()
        # self.memory.reset()
        # duration = 0
        # done = False
        
        # while not done:
        #     action = self.act(state, mode='train')
        #     x = self.memory.get()

        #     with torch.no_grad():
        #         logits = self.policyEstimator(x)
        #         dist = torch.distributions.Categorical(logits=logits)
        #         logps = dist.log_prob(torch.tensor(action, device=self.device)) 
                
        #     state, reward, terminated, truncated, _ = env.step(action)
        #     done = terminated or truncated
            
        #     states.append(torch.tensor(x, dtype=torch.uint8))
        #     actions.append(action)
        #     rewards.append(reward)
        #     logProbabilities.append(logps)
        #     duration += 1
        
        data = ...
        duration = ...
        
        # data = (states, actions, rewards, logProbabilities)
        return data, duration

def train(self):
    print("Start training")
    optimizer = torch.optim.Adam(self.valueEstimator.parameters(), lr=self.learningRate)
    
    rewardMemory = np.zeros((self.episodes), dtype=np.float32)

    for episode in range(self.episodes):
        if self.debug:
            print(f"Episode [{episode}/{self.episodes}]: Start")
        data, duration = self.runEpisode()
        if self.debug:
            print(f"Done after {duration} time steps")
        states, actions, rewards, old_logps = data
        returns = self.computeReturns(rewards, duration)
        
        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.uint8, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        old_logps = torch.stack(old_logps).detach().to(self.device)
        
        rewardMemory[episode] = sum(rewards)
        
        new_values = self.valueEstimator(states)
        valueLoss = F.mse_loss(new_values.squeeze(-1), returns)
        if self.debug:    
            print(f"{valueLoss}")
        
        #Jump start value estimator
        if episode < 40:
            optimizer.zero_grad()
            valueLoss.backward()
            optimizer.step()
            if self.debug:
                print("Warmup step, skip policy update")
            else:
                episodeMetric = rewardMemory[0:episode].mean()
                print(f"{episode}:{episodeMetric:.3f},{valueLoss:.3f}")
            continue      
        
        if self.debug:
            print("Computes advantages...", end=' ')
        advantages = self.computeAdvantage(states, returns)
        if self.debug:
            print("Done\nComputing surrogate loss...", end=' ')
        
        new_logits = self.policyEstimator(states)
        new_distribution = torch.distributions.Categorical(logits=new_logits)
        surrogateAdvantage = self.computedSurrogateAdvantage(
            new_distribution, 
            actions, 
            advantages, 
            old_logps
        )
        
        if self.debug:
            print(f'{surrogateAdvantage}\nValue loss... ', end=' ')
        
        if self.debug:
            print("Computing update step...", end=' ')
        with torch.no_grad():
            old_logits = self.policyEstimator(states)
            old_dist = torch.distributions.Categorical(logits=old_logits)
        
        self.policyEstimator.zero_grad()
        gradient = torch.autograd.grad(
            surrogateAdvantage, 
            self.policyEstimator.parameters(), 
            retain_graph=True
        )
        g = vectorize(gradient)
        
        stepDir = self.conjugateGradient(states, old_dist, g)
        stepSizeDen = torch.dot(stepDir, self.FVP(stepDir, states, old_dist))
        stepSize = torch.sqrt(2 * self.delta /(stepSizeDen + 1e-8))
        stepSize = torch.clamp(stepSize, 0.0, 1.0)
        fullStep = stepDir * stepSize
        if self.debug:
            print("Done")

        # Line search
        if self.debug:
            print("Starting line search")
        currentParams = vectorize(self.policyEstimator.parameters())
        success = False
        stepFraction = 1
        for _ in range(self.maxBacktrack):
            candidateParams = currentParams + stepFraction * fullStep
            self.setParameters(candidateParams)
            
            # Compute new surrogate and KL
            with torch.no_grad():
                new_logits = self.policyEstimator(states)
                new_dist = torch.distributions.Categorical(logits=new_logits)
                new_surrogateAdvantage = self.computedSurrogateAdvantage(
                    new_dist, 
                    actions, 
                    advantages, 
                    old_logps
                )

                divergence = torch.distributions.kl_divergence(old_dist, new_dist).mean()

            if self.debug:
                print(f"Improvement: {new_surrogateAdvantage - surrogateAdvantage}")
                print(f"Divergence: {divergence}")
            if new_surrogateAdvantage > surrogateAdvantage and divergence <= self.delta:
                if self.debug:
                    print(f"Line search successful, reduction factor: {stepFraction:.4f}")
                success = True
                break
            
            else:
                stepFraction /= 2
                if self.debug:
                    print(f"Candidate rejected, new fraction: {stepFraction}")
    
        if not success:
            self.setParameters(currentParams)
            if self.debug:
                print('Line search failed!')
        
        optimizer.zero_grad()
        valueLoss.backward()
        optimizer.step()
        
        episodeMetric = rewardMemory[episode-20:episode].mean()
        
        if self.debug:
            print(f'Episode {episode}> Recent Mean Reward: {episodeMetric:.2f},Current Reward:{rewardMemory[episode]}, Value Loss: {valueLoss.item():.2f}, Divergence: {divergence.item():.6f}')
        else:
            print(f'{episode}>{episodeMetric:.3f},{rewardMemory[episode]},{valueLoss.item():.3f},{divergence.item():.3f}')
    return