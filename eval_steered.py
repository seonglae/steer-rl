import os, re, random, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm

# Fix seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
device = "mps" if torch.backends.mps.is_available() else device

# Load gemma-2-2b model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
llm = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it", output_hidden_states=True).to(device)
llm.eval()

# Load SAE
from sae_lens import SAE
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release="gemma-scope-2b-pt-res-canonical",
    sae_id="layer_20/width_16k/canonical",
)
sae = sae.to(device)

# Define PolicyNetwork
class PolicyNetwork(torch.nn.Module):
    def __init__(self, latent_dim, dict_size):
        super().__init__()
        self.linear = torch.nn.Linear(latent_dim, dict_size)
        self.activation = torch.nn.Tanh()
    def forward(self, x):
        return self.activation(self.linear(x))

LATENT_DIM = 2304
DICT_SIZE = 16384
policy_net = PolicyNetwork(LATENT_DIM, DICT_SIZE).to(device)
policy_net.eval()
checkpoint = torch.load("./checkpoints/best_policy.pt")
policy_net.load_state_dict(checkpoint["policy_state_dict"])

# Define batch-aware steering hook
def batch_steering_hook(policy_net, sae):
    class SteeringHook:
        def __init__(self, policy_net, sae):
            self.policy_net = policy_net
            self.sae = sae
            self.observation = None
            self.action = None
            self.log_prob = None
        def __call__(self, module, inputs):
            residual = inputs[0]  # (B, seq_len, hidden_dim)
            obs = residual[:, -1, :]  # (B, latent_dim)
            self.observation = obs.detach()
            mean = self.policy_net(obs)
            sigma = torch.ones_like(mean) * 0.1
            dist = torch.distributions.Normal(mean, sigma)
            action = dist.rsample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            self.action = action.detach()
            self.log_prob = log_prob.detach()
            steering = self.sae.decode(action)  # (B, hidden_dim)
            residual[:, -1, :] = residual[:, -1, :] + steering
            return (residual,)
    return SteeringHook(policy_net, sae)

# Define MMLUDataLoader for validation (no limit)
class MMLUDataLoader:
    def __init__(self, dataset, split, limit=None):
        self.data = dataset[split]
        if limit is not None:
            self.data = self.data.select(range(limit))
        self.n_samples = len(self.data)
        self.index = 0
    def get_batch(self, batch_size):
        batch = []
        for _ in range(batch_size):
            sample = self.data[self.index % self.n_samples]
            self.index += 1
            batch.append({
                "question": sample["question"],
                "choices": sample["choices"],
                "answer": sample["answer"]
            })
        return batch
    def __iter__(self):
        for sample in self.data:
            yield {
                "question": sample["question"],
                "choices": sample["choices"],
                "answer": sample["answer"]
            }

mmlu_dataset = load_dataset("cais/mmlu", "all")
val_loader = MMLUDataLoader(mmlu_dataset, split="validation")  # full validation set

results = []
correct = 0
total = 0
batch_size = 16
batch_samples = []

for sample in tqdm(val_loader, total=val_loader.n_samples, desc="Evaluating"):
    batch_samples.append(sample)
    if len(batch_samples) == batch_size:
        prompts = []
        ground_truths = []
        for samp in batch_samples:
            question = samp["question"]
            choices = samp["choices"]
            if isinstance(samp["answer"], int):
                ground_truth = chr(65 + samp["answer"])
            else:
                ground_truth = samp["answer"].strip().upper()
            prompt = (question + "\n" +
                      "\n".join(f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)) +
                      "\nChoose one of the following options only: A, B, C, or D" +
                      "\nAnswer:")
            prompts.append(prompt)
            ground_truths.append(ground_truth)
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
        steering_hook = batch_steering_hook(policy_net, sae)
        hook_handle = llm.model.layers[20].register_forward_pre_hook(steering_hook)
        generated_ids = llm.generate(inputs, max_new_tokens=1)
        hook_handle.remove()
        for i in range(generated_ids.shape[0]):
            generated_token = tokenizer.decode(generated_ids[i, -1]).strip()
            predicted = generated_token[0].upper() if generated_token else ""
            results.append({"prompt": prompts[i], "ground_truth": ground_truths[i], "predicted": predicted})
            if predicted == ground_truths[i]:
                correct += 1
            total += 1
        batch_samples = []

if batch_samples:
    prompts = []
    ground_truths = []
    for samp in batch_samples:
        question = samp["question"]
        choices = samp["choices"]
        if isinstance(samp["answer"], int):
            ground_truth = chr(65 + samp["answer"])
        else:
            ground_truth = samp["answer"].strip().upper()
        prompt = (question + "\n" +
                  "\n".join(f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)) +
                  "\nChoose one of the following options only: A, B, C, or D" +
                  "\nAnswer:")
        prompts.append(prompt)
        ground_truths.append(ground_truth)
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
    steering_hook = batch_steering_hook(policy_net, sae)
    hook_handle = llm.model.layers[20].register_forward_pre_hook(steering_hook)
    generated_ids = llm.generate(inputs, max_new_tokens=1)
    hook_handle.remove()
    for i in range(generated_ids.shape[0]):
        generated_token = tokenizer.decode(generated_ids[i, -1]).strip()
        predicted = generated_token[0].upper() if generated_token else ""
        results.append({"prompt": prompts[i], "ground_truth": ground_truths[i], "predicted": predicted})
        if predicted == ground_truths[i]:
            correct += 1
        total += 1

accuracy = correct / total
print(f"Final MMLU Accuracy with Steering: {accuracy*100:.2f}%")
df = pd.DataFrame(results)
df.to_csv("gemma_2_2b_mmlu_steered_evaluation.csv", index=False)
