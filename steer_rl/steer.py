import torch

def batch_steering_hook(policy_net, sae):
    class SteeringHook:
        def __init__(self, policy_net, sae):
            self.policy_net = policy_net
            self.sae = sae
            self.observation = None   # will be tensor of shape (B, latent_dim)
            self.action = None        # (B,)
            self.log_prob = None      # (B,)

        def __call__(self, module, inputs):
            residual = inputs[0]  # shape: (B, seq_len, hidden_dim)
            observation = residual[:, -1, :]  # (B, latent_dim)
            self.observation = observation.detach()
            action, log_prob = select_action(self.policy_net, observation)
            steering = sae.decode(action)  # shape: (B, hidden_dim)
            self.action = action.detach()
            self.log_prob = log_prob.detach()
            # Add the corresponding steering vector to the last token.
            residual[:, -1, :] = residual[:, -1, :] + steering
            return (residual)
    return SteeringHook(policy_net, sae)

def select_action(policy_net, observation):
    # Get activated logits from policy_net (B, dict_size)
    logits = policy_net(observation)
    # Add small random noise for diversity
    epsilon = torch.randn_like(logits) * 0.01
    logits_noisy = logits + epsilon
    # Select top-1 value and index from noisy logits
    topk_vals, topk_indices = torch.topk(logits_noisy, k=2, dim=-1)
    # Clamp top values to a minimum of 1 (values below 1 become 1)
    topk_vals = torch.clamp(topk_vals, min=50.0)
    # Create one-hot action vector with the clamped top value at the selected index
    action = torch.zeros_like(logits)
    action.scatter_(1, topk_indices, topk_vals)
    # Compute softmax probabilities from the original logits and get log probability of the chosen index
    probs = torch.softmax(logits, dim=-1)
    chosen_probs = torch.gather(probs, 1, topk_indices)
    log_prob = torch.log(chosen_probs).mean(dim=1)
    return action, log_prob