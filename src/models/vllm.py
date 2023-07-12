from typing import AsyncGenerator
from libs.vllm.vllm.engine.arg_utils import AsyncEngineArgs
from libs.vllm.vllm.engine.async_llm_engine import AsyncLLMEngine
from libs.vllm.vllm import SamplingParams
from src.model import Model

class VLLM(Model):
    def __init__(self, dir: str, weights: str):
        args = AsyncEngineArgs
        args.download_dir = dir
        args.model = weights
        self.engine = AsyncLLMEngine(args=args)

    async def generate(self, prompt: str, args) -> AsyncGenerator[str, None]:
        params = SamplingParams(temperature=args.temparatue, top_p=args.top_p, top_k=args.top_k, max_tokens=args.max_tokens)
        generator = self.engine.generate(prompt, params)
        async for output in generator:
            yield output.outputs[0].text