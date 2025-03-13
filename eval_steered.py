import os, re, random, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
from steer_rl.ppo import PolicyNetwork 
from steer_rl.dataset import MMLUDataLoader
from steer_rl.steer import batch_steering_hook

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


LATENT_DIM = 2304
DICT_SIZE = 16384
policy_net = PolicyNetwork(LATENT_DIM, DICT_SIZE).to(device)
policy_net.eval()
checkpoint = torch.load("./checkpoints/best_policy.pt")
policy_net.load_state_dict(checkpoint["policy_state_dict"])

mmlu_dataset = load_dataset("cais/mmlu", "all")
test_loader = MMLUDataLoader(mmlu_dataset, split="test")  # full test set

results = []
correct = 0
total = 0
batch_size = 16
batch_samples = []

for sample in tqdm(test_loader, total=test_loader.n_samples, desc="Evaluating"):
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
