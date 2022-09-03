from numpy import array
from numpy import random

from util.movielens_dataset import MovielensDataset


class EmbeddingLookup:
    def __init__(self, embedding_size: int, key: str):
        self.embedding_size = embedding_size
        self.embeddings: dict[int, array] = {}
        self.key = key

    def get_embedding(self, feature) -> array:
        if feature not in self.embeddings.keys():
            self.embeddings[feature] = random.normal(loc=0.0, scale=1.0, size=self.embedding_size)
        return self.embeddings[feature]

    def update_embedding(self, feature, embedding: list[float]):
        self.embeddings[feature] = array(embedding)

    def get_embeddings(self, datasets: list[MovielensDataset]) -> (list[list[float]], dict):
        embedding = []
        embedding2idx = {}
        for dataset in datasets:
            for i in range(dataset.__len__()):
                features, _ = dataset.__getitem__(i)
                feature = features[self.key]
                if feature not in embedding2idx.keys():
                    embedding2idx[feature] = len(embedding)
                    embedding.append(self.get_embedding(feature))
        return embedding, embedding2idx

    def update_embeddings(self, embeddings: list[list[float]], embedding2idx: dict):
        for feature, index in embedding2idx.items():
            self.update_embedding(feature, embeddings[index])
