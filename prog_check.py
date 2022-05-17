import argparse

def str2bool(v: str):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')

def main(args):
    print(args)
    return 0

def parse_args():
    parser = argparse.ArgumentParser(description="test_args")
    parser.add_argument('-a','--A',default=1,type=int)
    parser.add_argument('-b','--B',default=False,type=str2bool)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
