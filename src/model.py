from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncGenerator, Generator

@dataclass
class GeneratorArgs:
    temperature: float = 1
    top_k: float = -1
    top_p: float = 1
    max_tokens: int = 256

class Model(ABC):
    @abstractmethod
    def __init__(self, dir: str) -> None:
        pass
    
    @abstractmethod
    def generate(self, prompt: str, id: str, args: GeneratorArgs = GeneratorArgs()) -> Generator[str, None, None] | AsyncGenerator[str, None]:
        pass

    @abstractmethod
    def abort(self, id: str) -> None:
        pass
    