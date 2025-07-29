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
                chosen =  tokenizer.apply_chat_template(
                            example["prompt"] + example["chosen"], tools=None, tokenize=False
                        )
                rejected =  tokenizer.apply_chat_template(
                            example["prompt"] + example["rejected"], tools=None, tokenize=False
                        )

                self.data.append({
                    "chosen": chosen,
                    "rejected": rejected
                })
        self.tokenizer = tokenizer
        self.max_len = tokenizer.model_max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chosen = self.tokenizer(self.data[idx]['chosen'], truncation=True, max_length=self.max_len, return_tensors='pt')
        rejected = self.tokenizer(self.data[idx]['rejected'], truncation=True, max_length=self.max_len, return_tensors='pt')
        # squeeze掉batch维
        chosen = {k:v.squeeze(0) for k,v in chosen.items()}
        rejected = {k:v.squeeze(0) for k,v in rejected.items()}
        return chosen, rejected


def dpo_collate(batch):
    # batch is [(chosen_inputs, rejected_inputs), ...]
    chosen_input_ids = [item[0]["input_ids"].squeeze() for item in batch]
    chosen_attention_mask = [item[0]["attention_mask"].squeeze() for item in batch]
    rejected_input_ids = [item[1]["input_ids"].squeeze() for item in batch]
    rejected_attention_mask = [item[1]["attention_mask"].squeeze() for item in batch]

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
    }