from abc import ABC, abstractmethod

class Packer(ABC):
    @abstractmethod
    def pack(self):
        pass
