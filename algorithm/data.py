import json
import torch
from torch.utils.data import Dataset
from collections import Counter
from typing import Dict, List, Tuple

class MemeDataset(Dataset):
    def __init__(self, data:List[Tuple[str, str, torch.Tensor]]):
        """
        data: List[(sentence_token_ids: torch.Tensor, label: torch.Tensor)]
        sentence_token_ids: torch.Tensor, needs to be padded
        label: torch.Tensor with shape of (20,)

        All data is loaded into memory, since there is not very much of them.
        Maybe try ragged tensor for irregular sizes.
        """
        self.idx = [d[0] for d in data]
        self.data = [d[1] for d in data]
        self.labels = torch.stack([d[2] for d in data])

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, idx):
        return self.idx[idx], self.data[idx], self.labels[idx]

class MemeEDataset(Dataset):
    def __init__(self, data:List[Tuple[str, str, torch.Tensor, torch.Tensor]]):
        """
        data: List[(sentence_token_ids: torch.Tensor, label: torch.Tensor, embedding: torch.Tensor)]
        sentence_token_ids: torch.Tensor, needs to be padded
        label: torch.Tensor, with shape of (20,)
        embedding: torch.Tensor, with shape of (100,), mean of embeddings of entities

        All data is loaded into memory, since there is not very much of them.
        Maybe try ragged tensor for irregular sizes.
        """
        self.idx = [d[0] for d in data]
        self.data = [d[1] for d in data]
        self.labels = torch.stack([d[2] for d in data])
        self.embeddings = torch.stack([d[3] for d in data])

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, idx):
        return self.idx[idx], self.data[idx], self.labels[idx], self.embeddings[idx]

def get_dict_data(data:List[Dict]) -> Dict[str,Tuple[List[str],str]]:
    return {
        d['id']: (d['labels'], d['text']) for d in data
    }
def get_list_data(data:Dict[str,Tuple[List[str],str]]) -> List[Dict]:
    return list(
        {'id': idx, 'labels': label, 'text': sent}
        for idx,(label,sent) in data.items()
    )
def get_data():
    train_data_file = 'data/training_set_task1.txt'
    dev_data_file = 'data/dev_set_task1.txt'
    test_data_file = 'data/test_set_task1.txt'

    # hard coded path
    with open(train_data_file, encoding='utf-8') as f:
        tr_data = get_dict_data(json.load(f))
    with open(dev_data_file, encoding='utf-8') as f:
        dev_data = get_dict_data(json.load(f))
    with open(test_data_file, encoding='utf-8') as f:
        test_data = get_dict_data(json.load(f))
    
    return tr_data, dev_data, test_data

def get_metadata():
    # use training data only
    tr_data, _, _ = get_data() # that's a huge waste lol, should've been done offline
    label_counter = Counter()
    for labels,_ in tr_data.values():
        label_counter.update(labels)
    label_counter = dict(label_counter)
    index_label = {i:label for i,label in enumerate(label_counter)}
    label_index = {label:i for i,label in index_label.items()}
    return label_counter, index_label, label_index

def get_embedding():
    with open('data/entity_embed.json') as f:
        return json.load(f)

def get_aug_data(filepath):
    with open(filepath, encoding='utf-8') as f:
        return json.load(f)

# I don't think it will be helpful to calculate embedding of labels in whatever methods.
LABEL_FREQ, INDEX_LABEL, LABEL_INDEX = get_metadata()
CATEGORY_NUM = 20
