from abc import ABC, abstractmethod
from typing import Generator

class Model(ABC):
    @abstractmethod
    def __init__(self, dir: str) -> None:
        pass
    
    @abstractmethod
    def generate(self, prompt: str) -> Generator[str, None, None]:
        pass
    