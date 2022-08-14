from numpy import ndarray
from numpy import random

from util.movielens_dataset import MovielensDataset


class EmbeddingLookup:
    def __init__(self, embedding_size: int, key: str):
        self.embedding_size = embedding_size
        self.embeddings: dict[int, ndarray] = {}
        self.key = key

    def get_embedding(self, feature) -> ndarray:
        if feature not in self.embeddings.keys():
            self.embeddings[feature] = random.normal(loc=0.0, scale=1.0, size=self.embedding_size)
        return self.embeddings[feature]

    def update_embedding(self, feature, embedding: list[float]):
        self.embeddings[feature] = ndarray(embedding)

    def get_embeddings(self, datasets: list[MovielensDataset]) -> (list[list[float]], dict):
        embedding = []
        embedding_map = {}
        for dataset in datasets:
            for i in range(dataset.__len__()):
                features, _ = dataset.__getitem__(i)
                feature = features[self.key]
                if feature not in embedding_map.keys():
                    embedding_map[feature] = len(embedding)
                    embedding.append(self.get_embedding(feature))
        return embedding, embedding_map

    def update_embeddings(self, embeddings: list[list[float]], embedding_map: dict):
        for feature, index in embedding_map.items():
            self.update_embedding(feature, embeddings[index])
