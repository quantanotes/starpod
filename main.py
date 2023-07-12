import sys
from src.cli import CLI

if __name__ == '__main__':
    sys.path.append("libs/exllama")
    sys.path.append("libs/vllm")
    CLI().parse_args() 
