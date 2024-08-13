import random
import pandas as pd
import numpy as np
from torch.utils.data import Dataset,DataLoader
import torch
from sklearn.model_selection import train_test_split
import json
from datasets import load_dataset,Features, Value
import copy
class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=1024, prompt_max_len=512, answer_max_len=512, seed=42):
        super().__init__()
        IGNORE_INDEX = -100
        self.max_length = max_length
        self.prompt_max_len = prompt_max_len
        self.answer_max_len = answer_max_len
        self.tokenizer = tokenizer
        self.input_ids = []
        self.labels = []
        self.attention_mask = []
        # 指定自定义字段
        features = Features({
            'prompt': Value('string'),
            'answer': Value('string')
        })
        sft_dataset = load_dataset('json', data_files=data_path, features=features)
        data = []
        # 遍历数据集并取出每个元素
        for example in sft_dataset['train']:
            prompt = example['prompt']
            answer = example['answer']
            messages = [
                {"role": "user", "content": prompt}
            ]
            prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            answer_text = answer + tokenizer.eos_token
            
            prompt_id = self.tokenizer.encode(prompt_text)
            if (len(prompt_id) > self.prompt_max_len):
                prompt_id = prompt_id[:self.prompt_max_len]

            answer_id = tokenizer.encode(answer_text)
            if (len(answer_id) > self.answer_max_len):
                answer_id = prompt_id[:self.prompt_max_len]
            input_id = prompt_id + answer_id
            labels = [self.tokenizer.pad_token_id] * len(prompt_id) + answer_id
            pad_len = self.max_length - len(input_id)
            input_id = input_id + [self.tokenizer.pad_token_id] * pad_len
            labels = labels + [self.tokenizer.pad_token_id] * pad_len
            labels = [(l if l != self.tokenizer.pad_token_id else IGNORE_INDEX ) for l in labels]
            input_id = torch.LongTensor(input_id)
            labels = torch.LongTensor(labels)
            attention_mask = input_id.ne(self.tokenizer.pad_token_id)
            data.append({
                "input_ids": input_id,
                "labels": labels,
                "attention_mask": attention_mask
            })

            # 打乱数据集
        random.seed(seed)
        random.shuffle(data)

        for item in data:
            self.input_ids.append(item["input_ids"])
            self.labels.append(item["labels"])
            self.attention_mask.append(item["attention_mask"])


    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, i: int):
       return {
            "input_ids": self.input_ids[i],
            "labels": self.labels[i],
            "attention_mask": self.attention_mask[i],
        }

