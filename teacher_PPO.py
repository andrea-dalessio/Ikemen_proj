import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Policy(nn.Module):
    
    def __init__(self, obs_dim, device=('cuda' if torch.cuda.is_available() else 'cpu')):
        super(Policy, self).__init__()
        self.device = device
        self.gamma = 0.99  # discount factor
        self.obs_dim = obs_dim
        
        self.policy_net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim)
        ).to(device)
        
        self.value_net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ).to(device)
        
    def forward(self, x):
        logits = self.policy_net(x)
        values = self.value_net(x).squeeze(-1)
        return logits, values
    
    def act(self, state):
            
        x = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            logits, _ = self.forward(x)

        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()

        return action.item() # idx of the action

    def train(self):
        env = ... # TODO: Initialize your environment here
        n_episodes = 800
        optimizer_value = torch.optim.Adam(self.value_net.parameters(), lr=1e-4)
        optimizer_policy = torch.optim.Adam(self.policy_net.parameters(), lr=3e-4)
        reward_history = []
        self.best_reward = -1e9 # Needed to prevent collapse

        
        for episode in range(n_episodes):
            states   = []
            actions  = []
            rewards  = []
            logps    = []
                       
            state, _ = env.reset()
            self.initd = False
            self.last_state = None
            self.clip_eps = 0.2
            done = False
            episode_reward = 0 # Where do I update this?
            
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
            
            # Training loop
            ppo_epochs = 4
            entropy_coef = 0.01
            value_coef = 0.5

            for _ in range(ppo_epochs):
                new_logits, new_values = self.forward(states)
                new_dist = torch.distributions.Categorical(logits=new_logits)
                new_logps = new_dist.log_prob(actions)

                ratio = torch.exp(new_logps - logps)

                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(new_values.squeeze(-1), returns)

                entropy = new_dist.entropy().mean()

                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

                optimizer_policy.zero_grad()
                optimizer_value.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)

                optimizer_policy.step()
                optimizer_value.step()
            
            # Logging
            reward_history.append(episode_reward)
            if len(reward_history) > 20:
                reward_history.pop(0)
            avg_reward = np.mean(reward_history)

            if (episode + 1) % 20 == 0:
                self.save()
                print(f"Episode {episode+1}, Average Reward: {avg_reward}")
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
