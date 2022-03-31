import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig
from algorithm.data import CATEGORY_NUM

class Model(nn.Module):
    """
    Abstract class for models.
    """
    def __init__(self, device, model_name):
        super().__init__()
        self.device = device
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.hidden_size = AutoConfig.from_pretrained(model_name).hidden_size
    def forward(self, input):
        pass

class BaseModel(Model):
    def __init__(self, device, model_name):
        super().__init__(device, model_name)
        self.fc1 = nn.Linear(self.hidden_size, 64, bias=False)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 64, bias=False)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(64, CATEGORY_NUM, bias=False)

    def forward(self, input):
        encoded_input = self.tokenizer(input, padding=True, truncation=True, return_tensors='pt')
        encoded_input = encoded_input.to(self.device)
        embed = self.model(**encoded_input)
        embed = embed.pooler_output
        h1 = self.act1(self.fc1(embed))
        h2 = self.act2(self.fc2(h1))
        return self.fc3(h2)

class Basev2Model(Model):
    def __init__(self, device, model_name):
        super().__init__(device, model_name)
        # so weird Sequential behave differently from components
        # self.seq = nn.Sequential(
        #     nn.Linear(self.hidden_size, 64, bias=False),
        #     nn.ReLU(),
        #     nn.Linear(64, 64, bias=False),
        #     nn.ReLU(),
        #     nn.Linear(64, CATEGORY_NUM, bias=False)
        # )
        self.fc1 = nn.Linear(self.hidden_size, 64, bias=False)
        self.act1 = nn.ReLU()
        self.bn = nn.BatchNorm1d(64)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(64, 64, bias=False)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(64, CATEGORY_NUM, bias=False)

    def forward(self, input):
        encoded_input = self.tokenizer(input, padding=True, truncation=True, return_tensors='pt')
        encoded_input = encoded_input.to(self.device)
        embed = self.model(**encoded_input)
        embed = embed.pooler_output
        h1 = self.act1(self.fc1(embed))
        b1 = self.bn(h1)
        h1 = self.drop1(h1)
        h2 = self.act2(self.fc2(h1)) + b1
        h2 = self.drop2(h2)
        return self.fc3(h2)
        # return self.seq(embed)

class EntityModel(Model):
    def __init__(self, device, model_name):
        super().__init__(device, model_name)
        # exactly same architecture with basev2
        self.fc1 = nn.Linear(self.hidden_size, 64, bias=False)
        self.act1 = nn.ReLU()
        self.bn = nn.BatchNorm1d(64)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(64, 64, bias=False)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(64, CATEGORY_NUM, bias=False)

        # embedding dim is 100
        self.fce1 = nn.Linear(self.hidden_size, 100, bias=False)
        self.acte1 = nn.LeakyReLU()
        self.drope1 = nn.Dropout(0.4)
        self.fce2 = nn.Linear(100, CATEGORY_NUM, bias=False)
    
    def forward(self, input, entity_embed):
        encoded_input = self.tokenizer(input, padding=True, truncation=True, return_tensors='pt')
        encoded_input = encoded_input.to(self.device)
        embed = self.model(**encoded_input)
        embed = embed.pooler_output
        
        h1 = self.act1(self.fc1(embed))
        b1 = self.bn(h1)
        h1 = self.drop1(h1)
        h2 = self.act2(self.fc2(h1)) + b1
        h2 = self.drop2(h2)

        rse = self.drope1(self.fce1(embed))
        h1e = self.acte1(rse * entity_embed)
        return self.fc3(h2) + self.fce2(h1e)

class Entityv2Model(Model):
    def __init__(self, device, model_name):
        super().__init__(device, model_name)
        self.fc1 = nn.Linear(self.hidden_size+100, 64, bias=False)
        self.act1 = nn.ReLU()
        self.bn = nn.BatchNorm1d(64)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(64, 64, bias=False)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(64, CATEGORY_NUM, bias=False)

    def forward(self, input,ent_emb):
        encoded_input = self.tokenizer(input, padding=True, truncation=True, return_tensors='pt')
        encoded_input = encoded_input.to(self.device)
        embed = self.model(**encoded_input)
        embed = embed.pooler_output
        h1 = self.act1(self.fc1(torch.cat([embed,ent_emb],1)))
        b1 = self.bn(h1)
        h1 = self.drop1(h1)
        h2 = self.act2(self.fc2(h1)) + b1
        h2 = self.drop2(h2)
        return self.fc3(h2)
