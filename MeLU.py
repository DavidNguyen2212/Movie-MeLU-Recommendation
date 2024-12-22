from typing import Any
import torch
from torch.nn import functional as F
from model.item import Item
from model.user import User
import torch.nn as nn
from torchtyping import TensorType


class Linear(nn.Linear):  # used in MAML to forward input with fast weight
    def __init__(self, in_features: int, out_features: int):
        super(Linear, self).__init__(in_features, out_features)
        self.weight.fast = None  # Lazy hack to add fast weight link
        self.bias.fast = None

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.linear(x, self.weight.fast, self.bias.fast)  # weight.fast (fast weight) is the temporarily adapted weight
        else:
            out = super(Linear, self).forward(x)
        return out


class User_preference_estimator(torch.nn.Module):
    def __init__(self, config: Any):
        super(User_preference_estimator, self).__init__()
        self.embedding_dim = config.embedding_dim
        self.fc1_in_dim = config.embedding_dim * 8
        self.fc2_in_dim = config.first_fc_hidden_dim
        self.fc2_out_dim = config.second_fc_hidden_dim

        self.item_emb = Item(config)
        self.user_emb = User(config)

        self.fc1 = Linear(self.fc1_in_dim, self.fc2_in_dim)
        self.fc2 = Linear(self.fc2_in_dim, self.fc2_out_dim)
        self.linear_out = Linear(self.fc2_out_dim, 1)
        self.final_part = nn.Sequential(
            self.fc1, 
            nn.ReLU(), # activation="relu"
            self.fc2, 
            nn.ReLU(), # activation="relu"
            self.linear_out
        )

    def forward(self, x: TensorType["batch_size", 10246], training: bool = True) -> TensorType["batch_size", 1]:
        rate_idx: TensorType["batch_size"] = x[:, 0]
        genre_idx: TensorType["batch_size", 25] = x[:, 1:26]
        director_idx: TensorType["batch_size", 2186] = x[:, 26:2212]
        actor_idx: TensorType["batch_size", 8029] = x[:, 2212:10242]
        gender_idx: TensorType["batch_size"] = x[:, 10242]
        age_idx: TensorType["batch_size"] = x[:, 10243]
        occupation_idx: TensorType["batch_size"] = x[:, 10244]
        area_idx: TensorType["batch_size"] = x[:, 10245]

        item_emb: TensorType["batch_size", "item_emb_dim"] = self.item_emb(rate_idx, genre_idx, director_idx, actor_idx)
        user_emb: TensorType["batch_size", "user_emb_dim"] = self.user_emb(gender_idx, age_idx, occupation_idx, area_idx)
        
        x: TensorType["batch_size", "fc1_in_dim"] = torch.cat((item_emb, user_emb), 1)
        x: TensorType["batch_size", 1] = self.final_part(x)
        return x