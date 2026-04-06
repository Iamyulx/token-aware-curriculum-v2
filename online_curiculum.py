import torch

class OnlineCurriculum:
    def __init__(self, model):
        self.model = model

    def compute_difficulty(self, input_ids):
        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            loss = outputs.loss
        return loss.item()

    def filter_batch(self, batch, threshold):
        filtered = []

        for sample in batch:
            difficulty = self.compute_difficulty(sample.unsqueeze(0))
            if difficulty < threshold:
                filtered.append(sample)

        return filtered
