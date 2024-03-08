import yaml
import torch


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Running on {device}')


if __name__ == '__main__':
    main()
