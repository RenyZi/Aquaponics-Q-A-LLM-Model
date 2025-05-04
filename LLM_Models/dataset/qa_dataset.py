import json
import torch
from torch.utils.data import Dataset

class QADataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        with open(data_path, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)  # expecting a list of dicts

        for entry in qa_data:
            context = entry["context"]
            question = entry["question"]
            answer_text = entry.get("answer", "")  # optional during inference

            # Encode inputs (only question and context used for prediction)
            encoded = tokenizer(
                question,
                context,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt"
            )

            input_ids = encoded["input_ids"].squeeze(0)
            attention_mask = encoded["attention_mask"].squeeze(0)

            self.samples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "question": question,
                "context": context,
                "answer": answer_text
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "input_ids": sample["input_ids"],
            "attention_mask": sample["attention_mask"],
            "question": sample["question"],
            "context": sample["context"],
            "answer": sample["answer"]  # optional, useful for comparison
        }
