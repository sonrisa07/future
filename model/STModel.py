from abc import abstractmethod

from model.QoSModel import QoSModel


class STModel(QoSModel):

    def __init__(self, k):
        super().__init__(k)

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

    @abstractmethod
    def get_tsp_data(self):
        pass

