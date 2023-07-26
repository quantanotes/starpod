import gc
from typing import AsyncGenerator
import torch
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
from src.model import Model, GeneratorArgs

class VLLM(Model):
    def __init__(self, dir: str, weights: str, tensor_parallel: int = 1):
        self._args = AsyncEngineArgs(weights, download_dir=dir, tensor_parallel_size=tensor_parallel)
        self._engine = AsyncLLMEngine.from_engine_args(self._args)

    async def generate(self, prompt: str, id: str, args: GeneratorArgs = GeneratorArgs()) -> AsyncGenerator[str, None]:
        params = SamplingParams(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, max_tokens=args.max_tokens)
        generator = self._engine.generate(prompt, params, id)
        prev = ""
        async for output in generator:
            curr = output.outputs[0].text
            delta = curr[len(prev):]
            prev = curr
            yield delta

    async def abort(self, id: str):
        await self._engine.abort(id)

    def reload(self):
        del self._engine
        torch.distributed.destroy_process_group()
        destroy_model_parallel()
        gc.collect()
        torch.cuda.empty_cache()
        self._engine = AsyncLLMEngine.from_engine_args(self._args)