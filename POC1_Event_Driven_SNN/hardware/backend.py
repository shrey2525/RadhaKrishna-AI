from abc import ABC, abstractmethod

class SNNBackend(ABC):
    @abstractmethod
    def compile(self, model):
        raise NotImplementedError

    @abstractmethod
    def run(self, events):
        raise NotImplementedError

    @abstractmethod
    def metrics(self):
        raise NotImplementedError
