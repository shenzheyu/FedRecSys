from numpy import ndarray
from numpy import random

from util.user_cluster_data import Sample


class EmbeddingLookup:
    def __init__(self, embedding_size):
        self.embedding_size = embedding_size
        self.embeddings: dict[int, ndarray] = {}

    def get_embedding(self, key: int) -> ndarray:
        if key not in self.embeddings.keys():
            self.embeddings[key] = random.normal(loc=0.0, scale=1.0, size=self.embedding_size)
        return self.embeddings[key]

    def update_embedding(self, key: int, embedding: list[float]):
        self.embeddings[key] = ndarray(embedding)

    def samples2embedding(self, samples: list[Sample]) -> (list[list[float]], dict[int, int]):
        embeddings = []
        embedding_map: dict[int, int] = {}
        index = 0
        for sample in samples:
            features = ['user_id/' + str(sample.user_id), 'gender/' + sample.gender,
                        'age/' + str(sample.age), 'occupation/' + str(sample.occupation),
                        'zip_code/' + sample.zip_code, 'movie_id/' + str(sample.movie_id),
                        'title/' + sample.title, 'genres/' + sample.genres,
                        'timestamp/' + str(sample.timestamp)]
            for feature in features:
                if hash(feature) in embedding_map.keys():
                    continue
                embeddings.append(self.get_embedding(hash(feature)).tolist())
                embedding_map[hash(feature)] = index
                index += 1
        return embeddings, embedding_map

    def update_embeddings(self, embeddings: list[list[float]], embedding_map: dict[int, int]):
        for key, index in embedding_map.items():
            self.update_embedding(key, embeddings[index])
