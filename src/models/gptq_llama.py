import os, glob
from typing import Generator
import torch 
from .exllama.model import ExLlama, ExLlamaCache, ExLlamaConfig
from .exllama.tokenizer import ExLlamaTokenizer
from .exllama.generator import ExLlamaGenerator
from model import Model

class GPTQLLaMa(Model):
    def __init__(self, dir: str):
        config_path = os.path.join(dir, "config.json")
        st_path = os.path.join(dir, "*.safetensors")
        tokeniser_path = os.path.join(dir, "tokenizer.model")

        config = ExLlamaConfig(config_path)
        config.model_path = glob.glob(st_path)[0]

        self._model = ExLlama(config)
        self._cache = ExLlamaCache(self._model)
        self._tokeniser = ExLlamaTokenizer(tokeniser_path)
        self._generator = ExLlamaGenerator(self._model, self._tokeniser, self._generator)

    def generate(self, prompt: str) -> Generator[str, None, None]:
        tokens = self._tokeniser.encode(prompt)
        self._generator.gen_begin(tokens)

        max_tokens = self._model.config.max_seq_len - tokens.shape[1]
        eos = torch.zeros((tokens.shape[0],), dtype = torch.bool)

        for _ in range(max_tokens):
            token = self._generator.gen_single_token()
            for i in range(token.shape[0]):
                if token[i, 0].item() == self._tokeniser.eos_token_id:
                    eos[i] = True
            if eos.all():
                break
            yield self._tokeniser.decode(token)