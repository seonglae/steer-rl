import torch
import torch.nn as nn
import torch.optim as optim

class PolicyNetwork(nn.Module):
    def __init__(self, latent_dim, dict_size):
        super(PolicyNetwork, self).__init__()
        self.linear = nn.Linear(latent_dim, dict_size)
        self.activation = nn.Tanh()  # using Tanh instead of ReLU
    
    def forward(self, obs):
        logits = self.linear(obs)
        activated = self.activation(logits)
        return activated

class CriticNetwork(nn.Module):
    def __init__(self, latent_dim):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(latent_dim, latent_dim)
        self.act = nn.Tanh()  # using Tanh here
        self.fc2 = nn.Linear(latent_dim, 1)
        
    def forward(self, obs):
        x = self.act(self.fc1(obs))
        value = self.fc2(x)
        return value


class PPOTrainer:
    def __init__(self, policy, critic, batch_size=8, ppo_clip=0.2, lr=1e-4):
        self.policy = policy
        self.critic = critic
        self.batch_size = batch_size
        self.ppo_clip = ppo_clip
        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr=lr)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr)
    
    def compute_advantages(self, rewards, values):
        return rewards - values

    def train_step(self, observations, actions, rewards, old_log_probs):
        # Compute policy outputs.
        mean = self.policy(observations)  # shape: (B, action_dim)
        sigma = torch.ones_like(mean) * 0.1  # fixed sigma
        dist = torch.distributions.Normal(mean, sigma)
        new_log_probs = dist.log_prob(actions).sum(dim=-1)
        
        # Compute critic outputs.
        values = self.critic(observations).squeeze(-1)
        
        advantages = rewards - values  # simple advantage
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        critic_loss = nn.MSELoss()(values, rewards)
        
        # Combine losses to perform a single backward pass.
        total_loss = policy_loss + critic_loss
        
        self.optimizer_policy.zero_grad()
        self.optimizer_critic.zero_grad()
        total_loss.backward()
        self.optimizer_policy.step()
        self.optimizer_critic.step()
        
        return policy_loss.item(), critic_loss.item()
