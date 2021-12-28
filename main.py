import argparse
from experiment import Experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['train', 'test'])
    args = parser.parse_args()
    seed = 42

    model = Experiment(seed)
    if args.action == 'train':
        model.train()
    elif args.action == 'test':
        model.test()


if __name__ == '__main__':
    main()
