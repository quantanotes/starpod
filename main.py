import sys
from src.cli import CLI

if __name__ == '__main__':
    sys.path.append("libs/exllama")
    CLI().parse_args() 
