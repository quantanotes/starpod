import os, glob
from typing import Generator
import torch 
from libs.exllama.model import ExLlama, ExLlamaCache, ExLlamaConfig
from libs.exllama.tokenizer import ExLlamaTokenizer
from libs.exllama.generator import ExLlamaGenerator
from src.logger import logger
from src.model import Model

class ExLLama(Model):
    def __init__(self, dir: str):
        config_path = os.path.join(dir, "config.json")
        st_path = os.path.join(dir, "*.safetensors")
        tokeniser_path = os.path.join(dir, "tokenizer.model")

        config = ExLlamaConfig(config_path)
        config.model_path = glob.glob(st_path)[0]

        logger.info("Instantiating GPTQ-LLaMa model")
        self._model = ExLlama(config)
        self._cache = ExLlamaCache(self._model)
        self._tokeniser = ExLlamaTokenizer(tokeniser_path)
        self._generator = ExLlamaGenerator(self._model, self._tokeniser, self._cache)

    def generate(self, prompt: str, id: str, args) -> Generator[str, None, None]:
        self._generator.settings = ExLlamaGenerator.Settings(args.temperature, args.top_k, args.top_p)

        tokens = self._tokeniser.encode(prompt)
        self._generator.gen_begin(tokens)

        max_tokens = args.max_tokens
        eos = torch.zeros((tokens.shape[0],), dtype = torch.bool)

        for _ in range(max_tokens):
            token = self._generator.gen_single_token()
            for i in range(token.shape[0]):
                if token[i, 0].item() == self._tokeniser.eos_token_id:
                    eos[i] = True
                    
            if eos.all():
                break

            decoded_token = self._tokeniser.decode(token)
            if isinstance(decoded_token, list):
                for item in decoded_token:
                    yield item
            else:
                yield decoded_token

    def abort(self, id: str):
        pass

    def reset(self):
        pass