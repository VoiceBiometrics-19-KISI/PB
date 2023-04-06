import torch


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def print_hi(name):
    print(f'Hi, {name}')


if __name__ == '__main__':
    print_hi('PyCharm')
    print(device)
