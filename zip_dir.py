import argparse
import subprocess
import shutil

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', help='Directory to zip')
    parser.add_argument('zip', type=str, default=None, help='Zip file to create')
    return parser.parse_args()

def main():
    args = parse_args()
    # zip all files in the directory, extract to current directory
    zipfile = args.dir if args.zip is None else args.zip
    shutil.make_archive(args.zip, 'zip', args.dir)

if __name__ == '__main__':
    main()