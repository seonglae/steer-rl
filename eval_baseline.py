import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm

# Set device.
device = "cuda" if torch.cuda.is_available() else "cpu"
device = "mps" if torch.backends.mps.is_available() else device

# Load gemma-2-2b model and tokenizer.
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
llm = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it", output_hidden_states=True).to(device)
llm.eval()

# Define a simple MMLUDataLoader that iterates over all test samples.
class MMLUDataLoader:
    def __init__(self, dataset, split):
        self.data = dataset[split]
        self.n_samples = len(self.data)
    def __iter__(self):
        for sample in self.data:
            # Expect each sample to have keys "question", "choices", and "answer".
            yield {
                "question": sample["question"],
                "choices": sample["choices"],
                "answer": sample["answer"]
            }

# Load the MMLU dataset (using the "cais/mmlu" dataset with the "all" configuration).
mmlu_dataset = load_dataset("cais/mmlu", "all")
test_loader = MMLUDataLoader(mmlu_dataset, split="test")

results = []
correct_count = 0
total = 0
batch_size = 16
batch_samples = []

# Process test samples in batches using tqdm for progress.
for sample in tqdm(test_loader, total=test_loader.n_samples, desc="Evaluating"):
    batch_samples.append(sample)
    if len(batch_samples) == batch_size:
        prompts = []
        ground_truths = []
        for samp in batch_samples:
            question = samp["question"]
            choices = samp["choices"]  # e.g., ["0", "4", "2", "6"]
            # Convert answer to a letter label (if answer is an integer, e.g., 1 -> "B").
            if isinstance(samp["answer"], int):
                ground_truth = chr(65 + samp["answer"])
            else:
                ground_truth = samp["answer"].strip().upper()
            prompt = (
                question + "\n" +
                "\n".join(f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)) +
                "\nChoose one of the following options only: A, B, C, or D" +
                "\nAnswer:"
            )
            prompts.append(prompt)
            ground_truths.append(ground_truth)
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
        # Generate one token per sample.
        generated_ids = llm.generate(inputs, max_new_tokens=1)
        for i in range(generated_ids.shape[0]):
            generated_token = tokenizer.decode(generated_ids[i, -1]).strip()
            predicted = generated_token[0].upper() if generated_token else ""
            results.append({
                "prompt": prompts[i],
                "ground_truth": ground_truths[i],
                "predicted": predicted
            })
            if predicted == ground_truths[i]:
                correct_count += 1
            total += 1
        batch_samples = []

# Process any remaining samples.
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
        prompt = (
            question + "\n" +
            "\n".join(f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)) +
            "\nChoose one of the following options only: A, B, C, or D" +
            "\nAnswer:"
        )
        prompts.append(prompt)
        ground_truths.append(ground_truth)
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
    generated_ids = llm.generate(inputs, max_new_tokens=1)
    for i in range(generated_ids.shape[0]):
        generated_token = tokenizer.decode(generated_ids[i, -1]).strip()
        predicted = generated_token[0].upper() if generated_token else ""
        results.append({
            "prompt": prompts[i],
            "ground_truth": ground_truths[i],
            "predicted": predicted
        })
        if predicted == ground_truths[i]:
            correct_count += 1
        total += 1

accuracy = correct_count / total
print(f"Final MMLU Accuracy: {accuracy*100:.2f}%")

# Save the results to a CSV file.
df = pd.DataFrame(results)
df.to_csv("gemma_2_2b_mmlu_evaluation.csv", index=False)
