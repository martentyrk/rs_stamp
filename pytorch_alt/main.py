import argparse


# 
def main(args):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch implementation of STAMP.')
    parser.add_argument('epochs', type=int, default=1)
    parser.add_argument('dataset', type=str, default='cikm16')
    args = parser.parse_args()
    main(args)