import torch

import torch
from torch.utils.data import Dataset, DataLoader
import json
from itertools import takewhile

class DPOPairDataset(Dataset):
    def __init__(self, jsonl_file, tokenizer):
        self.data = []
        with open(jsonl_file, 'r') as f:
            for line in f:
                example = json.loads(line)
                self.data.append(
                    example
                )
        self.tokenizer = tokenizer
        self.max_len = tokenizer.model_max_length

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        # tokenize prompt、chosen、rejected
        example = self.data[idx]
        prompt = self.tokenizer.apply_chat_template(example["prompt"], tools=None, tokenize=False, add_generation_prompt=True)
        prompt_toks = self.tokenizer(prompt, truncation=True, max_length=self.max_len, return_tensors='pt', add_special_tokens=False)
        prompt_length = prompt_toks["input_ids"].size(1)

        # 拼接full
        chosen_text = self.tokenizer.apply_chat_template(example["prompt"] + example["chosen"], tools=None, tokenize=False)
        rejected_text = self.tokenizer.apply_chat_template(example["prompt"] + example["rejected"], tools=None, tokenize=False)
        chosen = self.tokenizer(chosen_text, truncation=True, max_length=self.max_len, return_tensors='pt', add_special_tokens=False)
        rejected = self.tokenizer(rejected_text, truncation=True, max_length=self.max_len, return_tensors='pt', add_special_tokens=False)
        chosen = {k:v.squeeze(0) for k,v in chosen.items()}
        rejected = {k:v.squeeze(0) for k,v in rejected.items()}

        return chosen, rejected, prompt_length


def dpo_collate(batch):
    # batch is [(chosen_inputs, rejected_inputs, prompt_length), ...]
    chosen_input_ids = [item[0]["input_ids"].squeeze() for item in batch]
    chosen_attention_mask = [item[0]["attention_mask"].squeeze() for item in batch]
    rejected_input_ids = [item[1]["input_ids"].squeeze() for item in batch]
    rejected_attention_mask = [item[1]["attention_mask"].squeeze() for item in batch]
    prompt_lens = [item[2] for item in batch]  # new!

    # pad到同长度
    chosen_input_ids = torch.nn.utils.rnn.pad_sequence(chosen_input_ids, batch_first=True, padding_value=0)
    chosen_attention_mask = torch.nn.utils.rnn.pad_sequence(chosen_attention_mask, batch_first=True, padding_value=0)
    rejected_input_ids = torch.nn.utils.rnn.pad_sequence(rejected_input_ids, batch_first=True, padding_value=0)
    rejected_attention_mask = torch.nn.utils.rnn.pad_sequence(rejected_attention_mask, batch_first=True, padding_value=0)

    return {
        "chosen_input_ids": chosen_input_ids,
        "chosen_attention_mask": chosen_attention_mask,
        "rejected_input_ids": rejected_input_ids,
        "rejected_attention_mask": rejected_attention_mask,
        "prompt_lens": torch.tensor(prompt_lens, dtype=torch.long)
    }
