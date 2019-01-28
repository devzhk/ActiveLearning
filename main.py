from model import Model
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import numpy as np

from tensorboardX import SummaryWriter

model_savepath = 'save/model-01.pkl'
epoch_num = 10

def data_tf(x):
    x = x.resize((96, 96), 2)
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5
    x = x.transpose((2, 0, 1))
    x = torch.from_numpy(x)
    return x


if __name__ == '__main__':
    train_set = CIFAR10('data', train=True, transform=data_tf, download=False)
    train_data = DataLoader(train_set, batch_size=64, shuffle=True)
    test_set = CIFAR10('data', train=False, transform=data_tf, download=False)
    test_data = DataLoader(test_set, batch_size=64, shuffle=True)
    writer = SummaryWriter()
    model = Model(3)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    count = 0
    if torch.cuda.is_available():
        model = model.cuda()

    for epoch in range(epoch_num):
        train_loss = 0
        train_acc = 0
        model = model.train()
        for inputs, label in train_data:
            count += 1
            print(count)
            if torch.cuda.is_available():
                inputs = Variable(inputs.cuda())
                label = Variable(label.cuda())
            else:
                inputs = Variable(inputs)
                label = Variable(label)
            output = model(inputs)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            _, pred = output.max(1)
            num_correct = (pred == label).sum().item()
            acc = num_correct / inputs.shape[0]
            train_acc += acc

        eval_loss = 0
        eval_acc = 0
        model.eval()
        for inputs, label in test_data:
            if torch.cuda.is_available():
                inputs = Variable(inputs.cuda())
                label = Variable(label.cuda())
            else:
                inputs = Variable(inputs)
                label = Variable(label)
            output = model(inputs)
            loss = criterion(output, label)
            eval_loss += loss.item()
            _, pred = output.max(1)
            num_correct = (pred == label).sum().item()
            acc = num_correct / inputs.shape[0]
            eval_acc += acc

        print('epoch:{}, Train Loss:{.6f}, Train Acc:{.6f}, Eval Loss: {.6f}, Eval Acc:{.6f}'
              .format(epoch, train_loss / len(train_data), train_acc/len(train_data), eval_loss / len(test_data),
                      eval_acc / len(test_data)))
        writer.add_scalars('Loss', {'train': train_loss / len(train_data),
                                    'Eval': eval_loss / len(test_data)}, epoch)
        writer.add_scalars('Acc', {'Train': train_acc/len(train_data),
                                   'Eval': eval_acc}, epoch)

        torch.save(model.state_dict(), model_savepath)
        writer.close()

