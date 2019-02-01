from model import Model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.datasets import CIFAR10
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision.transforms import transforms
import numpy as np
import matplotlib.pyplot as plt
from cifar10_dataset import CIFAR10 as cif
import random
from tensorboardX import SummaryWriter
import resnet
model_savepath = 'save/model-al-top16-02-rush.pkl'
epoch_num = 320
sub_epoch_num = 20
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
learning_rate = 0.1
top_k = 64
batch_size = 16
threshold_acc = 0.93

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_transform = transforms.Compose(
        [transforms.Resize((96, 96), interpolation=2),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

test_transform = transforms.Compose(
    [transforms.Resize((96, 96), interpolation=2),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

def adjust_lr(optimizer, epoch):
    lr = learning_rate * (0.1 ** (epoch // 20))
    for pap in optimizer.param_groups:
        pap['lr'] = lr
    print('learning rate :{}'.format(lr))

def show(im):
    im = im / 2 + 0.5
    imnp = im.numpy()
    plt.imshow(np.transpose(imnp, (1, 2, 0)))
    plt.show()


def get_entropy(x):
    p = F.softmax(x, dim=1)
    lp = F.log_softmax(x, dim=1)
    return torch.mean(torch.sum(-torch.mul(p, lp), dim=1))


if __name__ == '__main__':
    init_set = cif('data', train=True, transform=transform_train, download=False)
    # train_data = DataLoader(init_set, batch_size=batch_size, shuffle=True)
    test_set = cif('data', train=False, transform=transform_test, download=False)
    test_data = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    train_idx = np.array([], dtype='int64')
    valid_idx = np.random.permutation(len(init_set))
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    writer = SummaryWriter()
    model = resnet.ResNet18()
    # model = Model(3)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        model.to(device)
        print('device:{}'.format(device))

    train_set = Subset(init_set, train_idx)
    train_data = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_set = Subset(init_set, valid_idx)
    valid_data = DataLoader(valid_set, batch_size=batch_size, shuffle=True)
    uncertain_list = []
    count = 0
    tic = 0
    # max_count = top_k * epoch_num

    for epoch in range(epoch_num):
        # adjust_lr(optimizer, epoch)
        model.eval()
        step = 8 + round(0.5 * count)
        # Valid stage and selection
        if epoch == 0 or tic == step:
            tic = 0
            print('training batch number:{}, train:{}, valid:{}'.format(count, len(train_set), len(valid_set)))
            for inputs, label, idx in valid_data:
                inputs, label = inputs.to(device), label.to(device)
                idx = idx.numpy()
                output = model(inputs)
                # selection
                # loss = criterion(output, label)
                loss = get_entropy(output)
                uncertain_list.append((loss.detach(), idx))
                # uncertain_list.append(idx)
            # uncertain_list = random.sample(uncertain_list, top_k)
            uncertain_list.sort(key=lambda x: x[0], reverse=True)
            for i in range(top_k):
                train_idx = np.concatenate((train_idx, uncertain_list[i][1]), axis=0)
                valid_idx = np.delete(valid_idx, uncertain_list[i][1])
            count += 1

            train_set = Subset(init_set, train_idx)
            train_data = DataLoader(train_set, batch_size=batch_size, shuffle=True)
            valid_set = Subset(init_set, valid_idx)
            valid_data = DataLoader(valid_set, batch_size=batch_size, shuffle=True)
            print('Unlabled pool max entropy:{:.4f}, min entropy:{:.4f}, delta:{:.4f}, Selection min entropy:{:.4f}'
                  .format(uncertain_list[0][0], uncertain_list[-1][0], uncertain_list[0][0]-uncertain_list[-1][0],
                          uncertain_list[top_k-1][0]))

            print('training batch number:{}, after selection train:{}, valid:{}'.format(count, len(train_set),
                                                                                        len(valid_set)))
            writer.add_scalars('Entropy', {'Selection Max': uncertain_list[0][0],
                                        'Selection Min': uncertain_list[top_k-1][0]}, count)
            uncertain_list = []
        # Train stage
        tic += 1
        train_loss = 0
        train_acc = 0
        model.train()
        # k = round((1 - count / max_count) * sub_epoch_num)
        # sub_train_acc = 0
        # sub_train_loss = 0
        # for e in range(k):
        for inputs, label, idx in train_data:
            inputs, label = inputs.to(device), label.to(device)

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
            # sub_train_acc = train_acc / len(train_data) - sub_train_acc
            # sub_train_loss = train_loss / len(train_data) - sub_train_loss
            # print('sub_epoch:{}, Train Loss:{:.6f}, Train Acc:{:.6f}'.format(e, sub_train_loss, sub_train_acc))

        # Test stage
        eval_loss = 0
        eval_acc = 0
        model.eval()
        for inputs, label, idx in test_data:
            inputs, label = inputs.to(device), label.to(device)

            output = model(inputs)
            loss = criterion(output, label)
            eval_loss += loss.item()
            _, pred = output.max(1)
            num_correct = (pred == label).sum().item()
            acc = num_correct / inputs.shape[0]
            eval_acc += acc

        etrain_loss = train_loss / len(train_data)
        etrain_acc = train_acc / len(train_data)
        eeval_loss = eval_loss / len(test_data)
        eeval_acc = eval_acc / len(test_data)

        print('epoch:{}, Train Loss:{:.6f}, Train Acc:{:.6f}, Eval Loss: {:.6f}, Eval Acc:{:.6f}'
              .format(epoch, etrain_loss, etrain_acc, eeval_loss,
                      eeval_acc))
        writer.add_scalars('Loss', {'train': etrain_loss,
                                    'Eval': eeval_loss}, epoch)
        writer.add_scalars('Acc', {'Train': etrain_acc,
                                   'Eval': eeval_acc}, epoch)
        if eeval_acc > threshold_acc:
            break

    torch.save(model.state_dict(), model_savepath)
    writer.close()

