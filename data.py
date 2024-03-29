from dataclasses import dataclass

import datasets
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
        # return input_ids, str(data['text_id'])
        return input_ids, str(data['semantic_ids'])
        


@dataclass
class IndexingCollator(DataCollatorWithPadding):
    def __call__(self, features):
        features = features[:500]
        input_ids = [{'input_ids': x[0]} for x in features]
        semantic_docids = [x[1] for x in features]
        inputs = super().__call__(input_ids)

        labels = self.tokenizer(
            semantic_docids, padding="longest", return_tensors="pt"
        ).input_ids

        # replace padding token id's of the labels by -100 according to https://huggingface.co/docs/transformers/model_doc/t5#training
        labels[labels == self.tokenizer.pad_token_id] = -100
        inputs['labels'] = labels
        return inputs


@dataclass
class QueryEvalCollator(DataCollatorWithPadding):
    def __call__(self, features):
        features = features[:500]
        input_ids = [{'input_ids': x[0]} for x in features]
        labels = [x[1] for x in features]
        inputs = super().__call__(input_ids)

        return inputs, labels


