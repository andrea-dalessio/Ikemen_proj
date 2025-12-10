import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Policy(nn.Module):
    continuous = False # you can change this

    # Defining some helper functions for TRPO
    def fisher_vector_product(self, v, states, old_dist, damping=1e-1):
        new_logits, _ = self.forward(states)
        new_dist = torch.distributions.Categorical(logits=new_logits)

        kl = torch.distributions.kl_divergence(old_dist, new_dist).mean()

        grads = torch.autograd.grad(
            kl,
            self.policy_net.parameters(),
            create_graph=True
        )

        flat_grads = self.flat_grad(grads)
        grad_v = torch.dot(flat_grads, v)

        grads2 = torch.autograd.grad(
            grad_v,
            self.policy_net.parameters()
        )

        flat_grads2 = self.flat_grad(grads2)

        return flat_grads2 + damping * v


    def flat_grad(self, grads): # Vector of gradients (one per param)
        return torch.cat([g.reshape(-1) for g in grads])

    def flat_params(self): # Vector of parameters
        return torch.cat([p.reshape(-1) for p in self.policy_net.parameters()])

    def conjugate_gradient(self, Fvp, b, max_iter=10, tol=1e-10): # Conjugate Gradient Algorithm (Solves Fv = b)
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)

        for i in range(max_iter):
            Fvp_p = Fvp(p)
            alpha = rdotr / (torch.dot(p, Fvp_p) + 1e-8)

            x = x + alpha * p
            r = r - alpha * Fvp_p
            new_rdotr = torch.dot(r, r)
            if new_rdotr < tol:
                break

            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr

        return x
    
    def set_flat_params(self, flat_params):
        index = 0
        with torch.no_grad():
            for param in self.policy_net.parameters():
                param_length = param.numel()
                param.copy_(flat_params[index:index + param_length].view(param.size()))
                index += param_length

    def __init__(self, device='cpu'):
        super(Policy, self).__init__()
        self.device = device
        self.k = 4  # number of stacked frames
        self.gamma = 0.99  # discount factor
        self.memory = np.zeros((self.k,) + (96, 96, 3), dtype=np.uint8) # Create stack shape and add a flag to check if stack is empty/full
        self.initd = False
        self.last_state = None # Has the ambient reset?
        
        self.policy_net = nn.Sequential(
            nn.Conv2d(self.k * 3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Linear(512, 5),  # 5 actions for discrete case
        ).to(device)
        
        self.value_net = nn.Sequential(
            nn.Conv2d(self.k * 3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        ).to(device)
        
    def forward(self, x):
        logits = self.policy_net(x)
        values = self.value_net(x).squeeze(-1)
        return logits, values
    
    def act(self, state):
        if self.last_state is not None:
            if np.allclose(state, self.last_state):
                self.initd = False

        self.last_state = state.copy()

        if not self.initd:
            for i in range(self.k):
                self.memory[i] = state
            self.initd = True
        else:
            self.memory[:-1] = self.memory[1:]
            self.memory[-1] = state
            
        # Data reshaping
        x = np.transpose(self.memory, (0, 3, 1, 2))
        x = x.reshape(-1, x.shape[2], x.shape[3])
        x = x.astype(np.float32) / 255.0            
        x = torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            logits, _ = self.forward(x)

        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()

        return action.item() # idx of the action

    def train(self):
        env = gym.make('CarRacing-v2', continuous=self.continuous)
        n_episodes = 800
        optimizer_value = torch.optim.Adam(self.value_net.parameters(), lr=1e-4)
        reward_history = []
        self.best_reward = -1e9 # Needed to prevent collapse

        
        for episode in range(n_episodes):
            states   = []
            actions  = []
            rewards  = []
            logps    = [] # Necessary for Fisher information matrix!
                       
            state, _ = env.reset()
            self.initd = False
            self.last_state = None
            self.delta = 0.006  # KL divergence limit
            done = False
            episode_reward = 0
            
            while not done: #Rollout!
                action = self.act(state)

                # Prepare tensors
                x = np.transpose(self.memory, (0, 3, 1, 2))
                x = x.reshape(-1, x.shape[2], x.shape[3])
                x = x.astype(np.float32) / 255.0            
                x = torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)

                with torch.no_grad():
                    logits, _ = self.forward(x)
                    dist = torch.distributions.Categorical(logits=logits) #Categorical for discrete actions
                    logp = dist.log_prob(torch.tensor(action, device=self.device))

                state, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                done = terminated or truncated

                # Saving to memory
                states.append(x.squeeze(0))
                actions.append(action)
                rewards.append(reward)
                logps.append(logp)

            # Compute returns
            returns = []
            G = 0.0
            
            for r in reversed(rewards):
                G = r + self.gamma * G
                returns.insert(0, G)
            
            # Tensorize everything
            states = torch.stack(states).to(self.device)
            actions = torch.tensor(actions, dtype=torch.int64, device=self.device)
            logps = torch.stack(logps).detach().to(self.device) # No gradients for log_pi,old!
            returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
            
            # Build advantages and normalize them
            _, values = self.forward(states)
            values = values.squeeze(-1)
            advantages = returns - values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Build surrogate loss and value loss
            new_logits, new_values = self.forward(states)
            new_dist = torch.distributions.Categorical(logits=new_logits)
            new_logps = new_dist.log_prob(actions)
            ratio = torch.exp(new_logps - logps)
            surrogate = (ratio * advantages).mean()
            
            # Loss computation for value network
            value_loss = F.mse_loss(new_values.squeeze(-1), returns)
            
            # WARM UP STEP: Just update the value network for a few steps to stabilize training
            if episode < 40:
                optimizer_value.zero_grad()
                value_loss.backward()
                optimizer_value.step()

                if (episode + 1) % 10 == 0:
                    print(f"[WARMUP] Episode {episode+1}, Value Loss: {value_loss.item():.4f}")

                continue # Skips ACTUAL TRPO during warm-up episodes            

            # KL, Fisher Information Matrix and Line Search to update policy
            new_logits_kl, _ = self.forward(states)
            new_dist_kl = torch.distributions.Categorical(logits=new_logits_kl)
            
            with torch.no_grad(): # Old policy is frozen
                old_logits, _ = self.forward(states)
                old_dist = torch.distributions.Categorical(logits=old_logits)
            
            self.policy_net.zero_grad()
            g = self.flat_grad(torch.autograd.grad(surrogate, self.policy_net.parameters(), retain_graph=True))
            step_dir = self.conjugate_gradient(lambda v: self.fisher_vector_product(v, states, old_dist),g)

            step_size = torch.sqrt(2 * self.delta /(torch.dot(step_dir, self.fisher_vector_product(step_dir, states, old_dist)) + 1e-8))
            step_size = torch.clamp(step_size, 0.0, 1.0) # Prevent too large steps
            full_step = step_dir * step_size
            
            # Temporary update for line search
            old_params = self.flat_params()
            success = False
            expected_improve = torch.dot(g, full_step)

            for step_frac in [0.5**i for i in range(10)]: # Backtracking line search, recompute each time it fails
                new_params = old_params + step_frac * full_step
                # Set new params
                self.set_flat_params(new_params)
                
                # Compute new surrogate and KL
                with torch.no_grad():
                    new_logits, _ = self.forward(states)
                    new_dist = torch.distributions.Categorical(logits=new_logits)
                    new_logps = new_dist.log_prob(actions)
                    ratio = torch.exp(new_logps - logps)
                    new_surrogate = (ratio * advantages).mean()
                    
                    new_logits_kl, _ = self.forward(states)
                    new_dist_kl = torch.distributions.Categorical(logits=new_logits_kl)
                    kl_new = torch.distributions.kl_divergence(old_dist, new_dist_kl).mean()

                actual_improve = new_surrogate - surrogate
                improve_ratio = actual_improve / (expected_improve + 1e-8)

                if improve_ratio > 0.01 and kl_new <= self.delta:
                    success = True
                    break
        
            if not success:
                self.set_flat_params(old_params)
                print('Line search failed!')

            # Update the value network
            optimizer_value.zero_grad()
            value_loss.backward()
            optimizer_value.step()
            
            # Logging
            reward_history.append(episode_reward)
            if len(reward_history) > 20:
                reward_history.pop(0)
            avg_reward = np.mean(reward_history)

            if (episode + 1) % 20 == 0:
                self.save()
                print(f'Episode {episode + 1}, Avg Reward: {avg_reward:.1f}, Value Loss: {value_loss.item():.4f}, KL after upd: {kl_new.item():.6f}, Step size: {step_size.item():.4f}\n')
                print(f'||g||: {torch.norm(g).item():.6f}, adv mean: {advantages.mean().item():.6f}, adv std: {advantages.std().item():.6f}')
                if avg_reward > self.best_reward:
                    self.best_reward = avg_reward
                    torch.save(self.state_dict(), "best_model.pt")
                    print("Best model saved!")
        return

    def save(self):
        torch.save(self.state_dict(), 'model.pt')

    def load(self):
        self.load_state_dict(torch.load('model.pt', map_location=self.device))

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret
