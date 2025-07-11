# =============================================================================
# data/federated.py 联邦数据分割
# =============================================================================
import numpy as np
from torch.utils.data import Subset
from collections import defaultdict


class FederatedDataset:
    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config = config
        self.targets = np.array(dataset.targets)
        self.client_indices = self._split_data()

    def _split_data(self):
        if self.config.non_iid:
            return self._non_iid_split()
        else:
            return self._iid_split()

    def _iid_split(self):
        indices = np.random.permutation(len(self.dataset))
        size = len(self.dataset) // self.config.num_clients
        return [indices[i * size:(i + 1) * size].tolist() for i in range(self.config.num_clients)]

    def _non_iid_split(self):
        num_classes = len(np.unique(self.targets))
        class_indices = defaultdict(list)

        for idx, label in enumerate(self.targets):
            class_indices[label].append(idx)

        client_indices = [[] for _ in range(self.config.num_clients)]

        for class_label in range(num_classes):
            indices = class_indices[class_label]
            np.random.shuffle(indices)

            # Dirichlet分布分配
            proportions = np.random.dirichlet([self.config.alpha] * self.config.num_clients)
            start = 0

            for client_id in range(self.config.num_clients):
                num_samples = int(len(indices) * proportions[client_id])
                if client_id == self.config.num_clients - 1:
                    end = len(indices)
                else:
                    end = start + num_samples
                client_indices[client_id].extend(indices[start:end])
                start = end

        return client_indices

    def get_client_data(self, client_id):
        indices = self.client_indices[client_id]
        return Subset(self.dataset, indices)
