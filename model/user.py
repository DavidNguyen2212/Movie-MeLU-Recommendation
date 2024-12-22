import torch
from torchtyping import TensorType
from typing import Any

class User(torch.nn.Module):
    def __init__(self, config: Any):
        super(User, self).__init__()
        self.num_gender = config.num_gender
        self.num_age = config.num_age
        self.num_occupation = config.num_occupation
        self.num_zipcode = config.num_zipcode
        self.embedding_dim = config.embedding_dim

        self.embedding_gender = torch.nn.Embedding(
            num_embeddings=self.num_gender,
            embedding_dim=self.embedding_dim
        )

        self.embedding_age = torch.nn.Embedding(
            num_embeddings=self.num_age,
            embedding_dim=self.embedding_dim
        )

        self.embedding_occupation = torch.nn.Embedding(
            num_embeddings=self.num_occupation,
            embedding_dim=self.embedding_dim
        )

        self.embedding_area = torch.nn.Embedding(
            num_embeddings=self.num_zipcode,
            embedding_dim=self.embedding_dim
        )

    def forward(self, 
        gender_idx: TensorType["batch_size", int], 
        age_idx: TensorType["batch_size", int], 
        occupation_idx: TensorType["batch_size", int], 
        area_idx: TensorType["batch_size", int]) -> TensorType["batch_size", "concat_dim"]:

        """
        The forward method of class User.
        Return a concatenated tensor of size ["batch_size", "concat_dim"],
        where concat_dim = embedding_dim * 4
        """

        # Simply create lookup table
        gender_emb = self.embedding_gender(gender_idx)
        age_emb = self.embedding_age(age_idx)
        occupation_emb = self.embedding_occupation(occupation_idx)
        area_emb = self.embedding_area(area_idx)

        return torch.cat((gender_emb, age_emb, occupation_emb, area_emb), 1)