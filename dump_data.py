import argparse
from helpers import get_pickle

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", metavar="data-file", type=str, help="data file")
    return parser.parse_args()

def main():
    args = parse_args()
    data = get_pickle(args.data_file)
    print(data)
    print(type(data))

if __name__ == "__main__":
    main()

