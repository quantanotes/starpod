import os
from typing import Optional
import argparse
from .download import download
from .logger import logger
from .pod import Pod

class CLI:
    def __init__(self):
        pass

    def parse_args(self):
        parser = argparse.ArgumentParser(description='Quanta LLM Pod CLI')
        subparsers = parser.add_subparsers(dest='command', help='Available commands')

        run_parser = subparsers.add_parser('run', help='Run a pod')
        run_parser.add_argument('model', type=str, help='Model name')
        run_parser.add_argument('weights', type=str, help='Weights')
        run_parser.add_argument('--tensor_parallel', '--tp', type=int, default=1)
        run_parser.add_argument('--quantisation', '--q', type=Optional[str], default=None)

        download_parser = subparsers.add_parser('download', help='Download weights from hugging face')
        download_parser.add_argument('name', help='Name of the hugging face repo to download from')

        args = parser.parse_args()

        match args.command:
            case 'run': self._run(args)
            case 'download': self._download(args)


    def _run(self, args: argparse.Namespace):
        model = None

        weights = args.weights
        dir = os.path.join(os.getcwd(), "weights")
        full_dir = os.path.join(dir, args.weights)

        if not os.path.exists(full_dir):
            logger.error(f'Weights {weights} are not installed')
            return

        tensor_parallel = args.tensor_parallel
        quantisation = args.quantisation

        match args.model:
            case 'test':
                from .models.test_model import TestModel
                model = TestModel(full_dir)
            case 'exllama':
                from .models.exllama_model import ExLLama
                model = ExLLama(full_dir)
            case 'vllm':
                from .models.vllm_model import VLLM
                model = VLLM(dir, weights, tensor_parallel, quantisation)
            case _:
                logger.error(f'No model named {args.model}')

        Pod(model)

    def _download(self, args):
        download(args.name) 
