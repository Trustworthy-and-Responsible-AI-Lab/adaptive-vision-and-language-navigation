import torch
import numpy as np
import torch.nn.functional as F


class LSH:
    """
    Implementation of Locality Sensitive Hashing for retrieval of cached embeddings.
    """
    def __init__(self, n_hyperplane, input_dim, sim_threshold=0.95):
        self.n_hyperplane = n_hyperplane
        self.input_dim = input_dim
        self.sim_threshold = sim_threshold
        self.hyperplanes = np.random.randn(n_hyperplane, input_dim)
        self.embeddings = {}

    def reset_for_new_episode(self):
        self.hyperplanes = np.random.randn(self.n_hyperplane, self.input_dim)
        self.embeddings.clear()

    def hash(self, vector):
        vector = vector.detach().cpu().numpy()
        sgn = self.hyperplanes.dot(vector)
        return tuple((sgn > 0).astype(int))

    def add_processed_embedding(self, raw_vector, processed_vector):
        hash_key = self.hash(raw_vector.reshape(-1))
        if hash_key not in self.embeddings:
            self.embeddings[hash_key] = []

        self.embeddings[hash_key].append((raw_vector.detach().cpu(), processed_vector.detach().cpu()))

    def get_similar_processed_embedding(self, raw):
        candidates = set()
        max_similarity = -1
        best_embedding = None

        hash_key = self.hash(raw.reshape(-1))
        bucket = self.embeddings.get(hash_key, [])
        for candidate_vector, processed_embedding in bucket:
            candidates.add((candidate_vector, processed_embedding))

        for candidate_vector, processed_embedding in candidates:
            similarity = torch.cosine_similarity(
                raw.cpu().flatten().float(),
                candidate_vector.flatten().float(),
                dim=0
            ).item()
                                
            if similarity > max_similarity:
                max_similarity = similarity
                best_embedding = processed_embedding

        if max_similarity >= self.sim_threshold:
            return best_embedding
        else:
            # print("no match")
            return None