import logging
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam
from typing import Dict, List, Tuple
from algorithm.data import MemeDataset, MemeEDataset, LABEL_FREQ, \
    INDEX_LABEL, LABEL_INDEX, get_embedding, CATEGORY_NUM
from algorithm.model import BaseModel, Basev2Model, EntityModel, Entityv2Model
from algorithm.preprocess import sent_filter

class Algorithm:
    """
    Abstract class for algorithms.
    """
    def __init__(self, config, model_name, device):
        self.model = None
        self.device = device
    
    def preprocess(self, data:Dict[str,Tuple[List[str],str]]) -> MemeDataset:
        pass

    def loss(self, pred:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        pass

    def predict(self, data:MemeEDataset) -> torch.Tensor:
        pass

    def train(self, data:MemeEDataset, 
                    dev_data:MemeEDataset, 
                    **params) -> None:
        total_size = len(data)
        dataloader = DataLoader(data, params['batch_size'], shuffle=True)
        optimizer = Adam(self.model.parameters(), params['lr'])
        logging.info('Start training')
        for i in range(params['epoch'] + 1):
            total_loss = 0
            for idx,input,label in dataloader:
                optimizer.zero_grad()
                pred = self.model(input)
                l = self.loss(pred,label)
                l.backward()
                optimizer.step()
                total_loss += l.item() * len(idx)
            total_loss /= total_size
            if i % 5 == 0: # too few sentences we have
                logging.info(f'epoch {i}, loss:{total_loss}')
                _, tl = self.predict(dev_data)
                logging.info(f'epoch {i}, dev_loss:{tl}')

class Baseline(Algorithm):
    def __init__(self, config, model_name, device='cpu'):
        super().__init__(config, model_name, device)
        self.loss_fn = nn.BCEWithLogitsLoss().to(device)
        self.model = Basev2Model(device, model_name).to(device)

    def preprocess(self, data:Dict[str,Tuple[List[str],str]]) -> MemeDataset:
        d = []
        for idx, (label_strs, sent) in data.items():
            label = torch.zeros(CATEGORY_NUM, dtype=torch.float)
            label[[LABEL_INDEX[label_str] for label_str in label_strs]] = 1.
            # TODO preprocess the sentence, i.e. filter, replace, ...
            d.append((idx,sent,label.to(self.device)))
        return MemeDataset(d)
    
    def loss(self, pred:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        return self.loss_fn(pred, target)

    def predict(self, data:MemeEDataset) -> Tuple[List[Dict], float]:
        dataloader = DataLoader(data)
        data = []
        total_loss = 0
        self.model.eval()
        with torch.no_grad():
            for idx,sents,true_label in dataloader:
                pred = self.model(sents)
                l = self.loss(pred, true_label).item()
                pred = torch.sigmoid(pred[0]).tolist()
                # the data order should preserve, otherwise we can only make the data more complex
                d = {}
                d['loss'] = l
                d['pred_scores'] = {
                    INDEX_LABEL[i]:score for i,score in enumerate(pred)
                }
                d['pred_labels'] = [INDEX_LABEL[i] for i,score in enumerate(pred) if score > 0.5]
                d['id'] = idx[0]
                data.append(d)
                total_loss += l
        self.model.train()
        return data, total_loss / len(data)

class Baseline_v2(Baseline):
    def __init__(self, config, model_name, device='cpu'):
        super().__init__(config, model_name, device)
        pos_weights = [LABEL_FREQ[INDEX_LABEL[i]] for i in range(CATEGORY_NUM)]
        pos_weights = torch.tensor(pos_weights)
        pos_weights = 1 / (torch.log(pos_weights) + 1)
        pos_weights /= pos_weights.min()
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weights).to(device)

class Baseline_v3(Baseline):
    def __init__(self, config, model_name, device='cpu'):
        super().__init__(config, model_name, device)

    def preprocess(self, data:Dict[str,Tuple[List[str],str]]) -> MemeDataset:
        d = []
        for idx, (label_strs, sent) in data.items():
            label = torch.zeros(CATEGORY_NUM, dtype=torch.float)
            label[[LABEL_INDEX[label_str] for label_str in label_strs]] = 1.
            #sent = sent.lower()
            sent = sent.replace('\n\n', '\n')
            if sent.endswith('\n'):
                sent = sent[:-1]
            d.append((idx,sent,label.to(self.device)))
        return MemeDataset(d)
    
    def preprocess_aug(self, data:Dict[str,Tuple[List[str],str]], 
                       aug_data:Dict[str,List[str]]) -> MemeDataset:
        d = []
        for idx, (label_strs, _) in data.items():
            label = torch.zeros(CATEGORY_NUM, dtype=torch.float)
            label[[LABEL_INDEX[label_str] for label_str in label_strs]] = 1.
            for sent in aug_data[idx]:
                d.append((idx,sent,label.to(self.device)))
        return MemeDataset(d)

class Entity_Algorithm(Algorithm):
    def __init__(self, config, model_name, device):
        super().__init__(config, model_name, device)
        self.loss_fn = nn.BCEWithLogitsLoss().to(device)
        self.model = Entityv2Model(device, model_name).to(device)
    
    def loss(self, pred:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        return self.loss_fn(pred, target)
    
    def preprocess(self, data:Dict[str,Tuple[List[str],str]]) -> MemeDataset:
        d = []
        embed_dict = get_embedding()
        for idx, (label_strs, sent) in data.items():
            label = torch.zeros(CATEGORY_NUM, dtype=torch.float)
            label[[LABEL_INDEX[label_str] for label_str in label_strs]] = 1.
            #sent = sent.lower()
            sent = sent.replace('\n\n', '\n')
            if sent.endswith('\n'):
                sent = sent[:-1]
            if idx in embed_dict:
                embed = torch.tensor(embed_dict[idx]).mean(axis=0)
            else:
                embed = torch.zeros(100)
            d.append((idx,sent,label.to(self.device),embed.to(self.device)))
        return MemeEDataset(d)

    def predict(self, data:MemeEDataset) -> Tuple[List[Dict], float]:
        dataloader = DataLoader(data)
        data = []
        total_loss = 0
        self.model.eval()
        with torch.no_grad():
            for idx,sents,true_label,ent_emb in dataloader:
                pred = self.model(sents,ent_emb)
                l = self.loss(pred, true_label).item()
                pred = torch.sigmoid(pred[0]).tolist()
                d = {}
                d['loss'] = l
                d['pred_scores'] = {
                    INDEX_LABEL[i]:score for i,score in enumerate(pred)
                }
                d['pred_labels'] = [INDEX_LABEL[i] for i,score in enumerate(pred) if score > 0.5]
                d['id'] = idx[0]
                data.append(d)
                total_loss += l
        self.model.train()
        return data, total_loss / len(data)

    def train(self, data:MemeEDataset, dev_data:MemeEDataset, **params) -> None:
        total_size = len(data)
        dataloader = DataLoader(data, params['batch_size'], shuffle=True)
        optimizer = Adam(self.model.parameters(), params['lr'])
        for i in range(params['epoch'] + 1):
            total_loss = 0
            for idx,input,label,ent_emb in dataloader:
                optimizer.zero_grad()
                pred = self.model(input,ent_emb)
                l = self.loss(pred,label)
                l.backward()
                optimizer.step()
                total_loss += l.item() * len(idx)
            total_loss /= total_size
            if i % 5 == 0: # too few sentences we have
                logging.info(f'epoch {i}, loss:{total_loss}')
                _, tl = self.predict(dev_data)
                logging.info(f'epoch {i}, dev_loss:{tl}')
