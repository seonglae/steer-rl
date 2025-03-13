
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
