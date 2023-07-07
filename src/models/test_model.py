from typing import Generator
from model import Model

class TestModel(Model):
    def __init__(self, dir: str) -> None:
        pass

    def generate(self, prompt: str) -> Generator[str, None, None]:
        for i in range(5):
            yield "ligma"
