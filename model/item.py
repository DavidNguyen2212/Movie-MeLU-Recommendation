from typing import Optional, Any
import torch
from torchtyping import TensorType

class Item(torch.nn.Module):
    def __init__(self, config: Any):
        super(Item, self).__init__()
        self.num_rate = config.num_rate
        self.num_genre = config.num_genre
        self.num_director = config.num_director
        self.num_actor = config.num_actor
        self.embedding_dim = config.embedding_dim

        self.embedding_rate = torch.nn.Embedding(
            num_embeddings=self.num_rate,
            embedding_dim=self.embedding_dim
        )

        self.embedding_genre = torch.nn.Linear(
            in_features=self.num_genre,
            out_features=self.embedding_dim,
            bias=False
        )

        self.embedding_director = torch.nn.Linear(
            in_features=self.num_director,
            out_features=self.embedding_dim,
            bias=False
        )

        self.embedding_actor = torch.nn.Linear(
            in_features=self.num_actor,
            out_features=self.embedding_dim,
            bias=False
        )

    def forward(self, 
        rate_idx: TensorType["batch_size", int], 
        genre_idx: TensorType["batch_size", "num_genre"], 
        director_idx: TensorType["batch_size", "num_director"], 
        actors_idx: TensorType["batch_size", "num_actor"], 
        vars: Optional[Any]=None) -> TensorType["batch_size", "concat_dim"]:
        """
        The forward method of class Item.
        Return a concatenated tensor of size ["batch_size", "concat_dim"],
        where concat_dim = embedding_dim * 4
        """
        # Embedding for rate (lookup table)
        rate_emb = self.embedding_rate(rate_idx)
        # Embedding (linear transformation and normalization)
        genre_emb = self.embedding_genre(genre_idx.float()) / torch.sum(genre_idx.float(), 1).view(-1, 1)
        director_emb = self.embedding_director(director_idx.float()) / torch.sum(director_idx.float(), 1).view(-1, 1)
        actors_emb = self.embedding_actor(actors_idx.float()) / torch.sum(actors_idx.float(), 1).view(-1, 1)

        return torch.cat((rate_emb, genre_emb, director_emb, actors_emb), 1)