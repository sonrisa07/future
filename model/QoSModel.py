from abc import abstractmethod


class QoSModel:

    def __init__(self, k):
        self.k = k

    @property
    @abstractmethod
    def net(self):
        pass

    @property
    @abstractmethod
    def edge_index(self):
        pass

    @property
    @abstractmethod
    def scaler(self):
        pass

    @abstractmethod
    def get_dataloaders(self, scope, split):
        pass
