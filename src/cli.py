import os
import argparse
from .download import download
from .logger import logger
from .pod import Pod

class CLI:
    def __init__(self):
        pass

    def parse_args(self):
        parser = argparse.ArgumentParser(description="Quanta LLM Pod CLI")
        subparsers = parser.add_subparsers(dest='command', help='Available commands')

        run_parser = subparsers.add_parser('run', help='Run a pod')
        run_parser.add_argument('model', type=str, help='Model name')
        run_parser.add_argument('weights', type=str, help='Weights')

        download_parser = subparsers.add_parser('download', help='Download weights from hugging face')
        download_parser.add_argument('name', help='Name of the hugging face repo to download from')

        args = parser.parse_args()

        match args.command:
            case "run": self._run(args)
            case "download": self._download(args)


    def _run(self, args: argparse.Namespace):
        model = None
        weights = os.path.join("./weights/", args.weights)
        if not os.path.exists(weights):
            logger.error(f"No weights at folder {weights}")
            
        match args.model:
            case "test":
                from .models.test_model import TestModel
                model = TestModel(weights)
            case "gptq-llama":
                from .models.gptq_llama import GPTQLLaMa
                model = GPTQLLaMa(weights)
            case _:
                logger.error(f"No model named {args.model}")

        Pod(model)

    def _download(self, args):
        download(args.name) 