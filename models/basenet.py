import logging
from abc import abstractmethod

# from gans.loaders import loader_factory

log = logging.getLogger("basenet")


class BaseNet(object):
    """
    Base model for segmentation neural networks
    """
    def __init__(self, conf):
        self.model = None
        self.conf = conf

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def save_models(self):
        pass

    @abstractmethod
    def load_models(self):
        pass
