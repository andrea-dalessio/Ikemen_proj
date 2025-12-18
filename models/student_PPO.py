import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json

with open('configs.json', mode='r') as f:
    CONFIGS = json.load(f)

class Network(nn.Module):
    def __init__(self, outputSize, device, stackSize=4):
        super().__init__()
        self.stackSize = stackSize
        self.device = device
        
        inputChannels = 3 * self.stackSize
        
        # 640 x 480
        
        # TODO Change the input to the one of ikemen
        
        self.conv1 = nn.Conv2d(inputChannels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.hidden = nn.Linear(4096, 512)
        self.output = nn.Linear(512, outputSize)
        
        self.act = nn.ReLU()

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        x = x.to(device=self.device)
        x = x.permute(0,3,1,2)
        x = x/255.0
        
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        
        x = x.reshape(x.size(0), -1)
        
        x = self.act(self.hidden(x))
        x = self.output(x)
        
        return x

class MemoryStack:
    def __init__(self, stackSize=4):
        self.data = np.zeros((stackSize, 96, 96, 3), dtype=np.uint8)
        self.size = stackSize
        self.used = 0
        self.current = 0
        
    def record(self, state):
        self.data[self.current] = state
        self.current = (self.current + 1) % self.size
        self.used = min(self.size, self.used + 1)
    
    def reset(self):
        self.data = np.zeros((self.size, 96, 96, 3), dtype=np.uint8)
        self.used = 0
        self.current = 0
        
    def get(self):
        indices = list(range(self.current, self.size)) + list(range(0, self.current))
        frames = [self.data[i] for i in indices]
        return np.concatenate(frames, axis=2)

    def ready(self):
        return self.used == self.size

def vectorize(params):
    return torch.cat([p.reshape(-1) for p in params])

class StudentModel(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
            deviceName = torch.cuda.get_device_name(torch.cuda.current_device())
            print(f"Using device: {deviceName}")
        else:
            self.device = torch.device('cpu')
            print(f"Using device: cpu")
        
        self.debug = False
        
        self.actionSpace = CONFIGS['']
        self.maxBacktrack = 15
        
        self.stackSize = 4
        self.gamma = 0.99
        self.delta = 0.006
        self.learningRate = 1e-4
        self.episodes = 800
        
        self.memory = MemoryStack(self.stackSize)
        
        policy_net = Network(self.actionSpace, self.device, stackSize=self.stackSize)
        value_net = Network(1, self.device, stackSize=self.stackSize)
        
        self.policyEstimator = policy_net.to(self.device)
        self.valueEstimator = value_net.to(self.device)

    def FVP(self, v, states, old_dist, damping=1e-1):
        newLogits = self.policyEstimator(states)
        newDistribution = torch.distributions.Categorical(logits=newLogits)

        divergence = torch.distributions.kl_divergence(old_dist, newDistribution).mean()

        grads_d1 = torch.autograd.grad(divergence, self.policyEstimator.parameters(), create_graph=True)

        vectorizeGradients_d1 = vectorize(grads_d1)
        grad_v = torch.dot(vectorizeGradients_d1, v)

        grads_d2 = torch.autograd.grad(grad_v, self.policyEstimator.parameters())

        vectorizeGradients_d2 = vectorize(grads_d2)

        return vectorizeGradients_d2 + damping * v

    def conjugateGradient(self, state, old_dist, b, maxIterations=10, tolerance=1e-10):
        x = torch.zeros_like(b)
        residual = b.clone()
        direction = b.clone()
        rdotr = torch.dot(residual, residual)

        for _ in range(maxIterations):
            Fvp_p = self.FVP(direction, state, old_dist)
            alpha = rdotr / (torch.dot(direction, Fvp_p) + 1e-8)

            x = x + alpha * direction
            residual = residual - alpha * Fvp_p
            new_rdotr = torch.dot(residual, residual)
            if new_rdotr < tolerance:
                break

            beta = new_rdotr / rdotr
            direction = residual + beta * direction
            rdotr = new_rdotr

        return x
    
    def setParameters(self, parametersVector):
        index = 0
        with torch.no_grad():
            for param in self.policyEstimator.parameters():
                n = param.numel()
                param.copy_(parametersVector[index:index + n].view(param.size()))
                index += n
    
    def act(self, state, mode='test'):
        if self.memory.ready():
            self.memory.record(state)
        else:
            for _ in range(self.stackSize):
                self.memory.record(state)
                
        networkInput = self.memory.get()
        
        with torch.no_grad():
            logits = self.policyEstimator(networkInput)

        if mode == 'train':
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
        else:
            action = logits.argmax(dim=-1)

        return action.item()


    def runEpisode(self, env):
        states   = []
        actions  = []
        rewards  = []
        logProbabilities = []
        
        state, _ = env.reset()
        self.memory.reset()
        duration = 0
        done = False
        
        while not done:
            action = self.act(state, mode='train')
            x = self.memory.get()

            with torch.no_grad():
                logits = self.policyEstimator(x)
                dist = torch.distributions.Categorical(logits=logits)
                logps = dist.log_prob(torch.tensor(action, device=self.device)) 
                
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            states.append(torch.tensor(x, dtype=torch.uint8))
            actions.append(action)
            rewards.append(reward)
            logProbabilities.append(logps)
            duration += 1
            
        data = (states, actions, rewards, logProbabilities)
        return data, duration
        
    def computeReturns(self, rewards, duration):
        returns = np.zeros((duration,))
        G = 0.0
        for t in reversed(range(duration)):
            G = rewards[t] + self.gamma * G
            returns[t] = G
        return returns

    def computeAdvantage(self, states, returns):
        values = self.valueEstimator(states)
        values = values.squeeze(-1)
        
        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages

    def computedSurrogateAdvantage(self, distribution, actions, advantages, old_logps):
        logProbabilities = distribution.log_prob(actions)
        surrogateAdvantage = torch.exp(logProbabilities - old_logps) * advantages
        return surrogateAdvantage.mean()

    def train(self):
        print("Start training")
        env = gym.make('CarRacing-v2', continuous=self.continuous)
        optimizer = torch.optim.Adam(self.valueEstimator.parameters(), lr=self.learningRate)
        
        rewardMemory = np.zeros((self.episodes), dtype=np.float32)

        for episode in range(self.episodes):
            if self.debug:
                print(f"Episode [{episode}/{self.episodes}]: Start")
            data, duration = self.runEpisode(env)
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

    def save(self):
        torch.save(self.state_dict(), 'model.pt')

    def load(self):
        self.load_state_dict(torch.load('model.pt', map_location=self.device))

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret
