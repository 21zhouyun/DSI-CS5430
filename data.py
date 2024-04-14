from dataclasses import dataclass
import re

import datasets
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, DataCollatorWithPadding

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, key):
        node = self.root
        for part in key:
            if part not in node.children:
                node.children[part] = TrieNode()
            node = node.children[part]
        node.is_end_of_word = True

    def search(self, key):
        node = self.root
        for part in key:
            if part not in node.children:
                return False
            node = node.children[part]
        return node.is_end_of_word

    def starts_with(self, prefix):
        node = self.root
        for part in prefix:
            if part not in node.children:
                return False
            node = node.children[part]
        return True

    def insert(self, tokens):
        current_node = self.root
        for token in tokens:
            if token not in current_node.children:
                current_node.children[token] = TrieNode()
            current_node = current_node.children[token]
        current_node.is_end_of_word = True

    def get_valid_first_tokens(self):
        # Return all children of the root node
        return list(self.root.children.keys())

    def get_valid_next_tokens(self, prefix):
        node = self.root
        for part in prefix:
            if part in node.children:
                node = node.children[part]
            else:
                return []
        return list(node.children.keys())



class IndexingTrainDataset(Dataset):
    def __init__(
            self,
            path_to_data,
            max_length: int,
            cache_dir: str,
            tokenizer: PreTrainedTokenizer,
    ):
        self.train_data = datasets.load_dataset(
            'json',
            data_files=path_to_data,
            ignore_verifications=False,
            cache_dir=cache_dir
        )['train']

        self.max_length = max_length
        self.tokenizer = tokenizer
        self.total_len = len(self.train_data)
        


    def __len__(self):
        return self.total_len
    
    def __getitem__(self, item):
        data = self.train_data[item]

        input_ids = self.tokenizer(data['text'],
                                   return_tensors="pt",
                                   truncation='only_first',
                                   max_length=self.max_length).input_ids[0]
        # print(input_ids, ": ",str(data['semantic_ids']))                          
        return input_ids, str(data['semantic_ids'])



@dataclass
class IndexingCollator(DataCollatorWithPadding):
    def preprocess_docids(self, docids):
        # Ensure docids is a string
        if not isinstance(docids, str):
            docids = str(docids) if docids is not None else ''
        # Insert spaces between "cXX" patterns
        spaced_docids = re.sub(r"(c\d+)", r"\1 ", docids).strip()
        return spaced_docids

    def tokenize_docids(self, docids_batch, tokenizer):
        # Tokenize each document ID in the batch after preprocessing
        tokenized_outputs = [tokenizer(
            self.preprocess_docids(docid),
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )['input_ids'].squeeze(0) for docid in docids_batch]
        # Stack all tokenized outputs into a single tensor
        tokenized_batch = torch.stack(tokenized_outputs, dim=0)
        return tokenized_batch

    # def get_semantic_ids(self, features):
    #     input_ids = [{'input_ids': x[0]} for x in features]
    #     semantic_docids = [x[1] for x in features]
    #     return semantic_docids

    def __call__(self, features):
        input_ids = [{'input_ids': x[0]} for x in features]
        semantic_docids = [x[1] for x in features]
        # print("input_ids: ",input_ids)
        # print("semantic_docids: ",semantic_docids)
        inputs = super().__call__(input_ids)
        # Tokenize all semantic docids in the batch
        labels = self.tokenize_docids(semantic_docids, self.tokenizer)
        # print("labels: ",labels)
        # Set padding token IDs in labels to -100
        labels[labels == self.tokenizer.pad_token_id] = -100
        inputs['labels'] = labels
        # print("inputs: ", inputs)
        return inputs


@dataclass
class QueryEvalCollator(DataCollatorWithPadding):
    def __call__(self, features):
        input_ids = [{'input_ids': x[0]} for x in features]
        labels = [x[1] for x in features]
        inputs = super().__call__(input_ids)

        return inputs, labels


