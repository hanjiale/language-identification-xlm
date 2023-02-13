import torch
import numpy as np
from torch.utils.data import Dataset
from util import get_args, get_labels, get_label_names
import os
import csv


class IDDataset(Dataset):

    def __init__(self, path=None, name=None, tokenizer=None, format="csv"):

        self.args = get_args()
        assert self.args.tuning_type == "ft"

        self.lang_labels = get_labels()
        self.label2id = {label: id for id, label in enumerate(self.lang_labels)}

        self.num_labels = len(self.lang_labels)
        self.tokenizer = tokenizer
        self.mask_token = tokenizer.mask_token
        self.pad_token_id = tokenizer.pad_token_id

        self.datas = []
        if format == "csv":
            with open(os.path.join(path, name), 'r', encoding="utf-8") as csvfile:
                reader = csv.reader(csvfile)
                for line in reader:
                    self.datas.append(line)
            self.datas = self.datas[1:]

        elif format == "txt":
            with open(os.path.join(path, name), 'r', encoding="utf-8") as file:
                for line in file.readlines():
                    self.datas.append(line)
        else:
            raise NotImplementedError("This file format is not yet supported!")

    def __getitem__(self, index):
        item = self.datas[index]
        if len(item) == 2:
            label, sentence = item
            input_ids = self.tokenizer.encode(sentence, max_length=self.args.max_seq_length, truncation=True)
            label_id = self.label2id[label]

            attention_mask = [1] * len(input_ids)
            padding_length = self.args.max_seq_length - len(input_ids)

            if padding_length > 0:
                input_ids = input_ids + ([self.pad_token_id] * padding_length)
                attention_mask = attention_mask + ([0] * padding_length)

            assert len(input_ids) == len(attention_mask) == self.args.max_seq_length

            input_ids = torch.tensor(np.array(input_ids)).long()
            attention_mask = torch.tensor(np.array(attention_mask)).long()
            label_id = torch.tensor(np.array(label_id)).long()

            return input_ids, attention_mask, label_id

        else:
            input_ids = self.tokenizer.encode(item, max_length=self.args.max_seq_length, truncation=True)

            attention_mask = [1] * len(input_ids)
            padding_length = self.args.max_seq_length - len(input_ids)

            if padding_length > 0:
                input_ids = input_ids + ([self.pad_token_id] * padding_length)
                attention_mask = attention_mask + ([0] * padding_length)

            assert len(input_ids) == len(attention_mask) == self.args.max_seq_length

            input_ids = torch.tensor(np.array(input_ids)).long()
            attention_mask = torch.tensor(np.array(attention_mask)).long()

            return input_ids, attention_mask


    def __len__(self):
        return len(self.datas)


class IDDataset_prompt(Dataset):

    def __init__(self, path=None, name=None, tokenizer=None, format="csv"):

        self.args = get_args()
        assert self.args.tuning_type == "pt"
        self.tokenizer = tokenizer
        self.mask_token = tokenizer.mask_token
        self.pad_token_id = tokenizer.pad_token_id

        self.lang_labels = get_labels()
        self.lang_names = get_label_names()
        self._get_label_names()

        self.label2id = {label: id for id, label in enumerate(self.lang_labels)}
        self.num_labels = len(self.lang_labels)

        self.datas = []
        if format == "csv":
            with open(os.path.join(path, name), 'r', encoding="utf-8") as csvfile:
                reader = csv.reader(csvfile)
                for line in reader:
                    self.datas.append(line)
            self.datas = self.datas[1:]

        elif format == "txt":
            with open(os.path.join(path, name), 'r', encoding="utf-8") as file:
                for line in file.readlines():
                    self.datas.append(line)
        else:
            raise NotImplementedError("This file format is not yet supported!")

    def _get_label_names(self):
        self.lang_names_ids = [self.tokenizer.encode(label_name, add_special_tokens=False) for label_name in self.lang_names]


    def __getitem__(self, index):
        item = self.datas[index]
        prompt = "The language of this sentence is " + self.mask_token
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(item) == 2:
            label, sentence = item
            input_ids = self.tokenizer.encode(sentence, truncation=True,
                                              max_length=self.args.max_seq_length - len(prompt_ids) - 2,
                                              add_special_tokens=False)
            input_ids = self.tokenizer.build_inputs_with_special_tokens(input_ids + prompt_ids)
            label_id = self.label2id[label]

            attention_mask = [1] * len(input_ids)
            padding_length = self.args.max_seq_length - len(input_ids)

            if padding_length > 0:
                input_ids = input_ids + ([self.pad_token_id] * padding_length)
                attention_mask = attention_mask + ([0] * padding_length)

            assert len(input_ids) == len(attention_mask) == self.args.max_seq_length

            input_ids = torch.tensor(np.array(input_ids)).long()
            attention_mask = torch.tensor(np.array(attention_mask)).long()
            label_id = torch.tensor(np.array(label_id)).long()

            return input_ids, attention_mask, label_id
        else:
            input_ids = self.tokenizer.encode(item, truncation=True,
                                              max_length=self.args.max_seq_length - len(prompt_ids) - 2,
                                              add_special_tokens=False)
            input_ids = self.tokenizer.build_inputs_with_special_tokens(input_ids + prompt_ids)

            attention_mask = [1] * len(input_ids)
            padding_length = self.args.max_seq_length - len(input_ids)

            if padding_length > 0:
                input_ids = input_ids + ([self.pad_token_id] * padding_length)
                attention_mask = attention_mask + ([0] * padding_length)

            assert len(input_ids) == len(attention_mask) == self.args.max_seq_length

            input_ids = torch.tensor(np.array(input_ids)).long()
            attention_mask = torch.tensor(np.array(attention_mask)).long()

            return input_ids, attention_mask

    def __len__(self):
        return len(self.datas)

