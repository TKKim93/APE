from __future__ import print_function
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from model.resnet import resnet34
from model.basenet import AlexNetBase, Predictor_latent, Predictor_deep_latent, grad_reverse
from utils.lr_schedule import inv_lr_scheduler
from utils.return_dataset import return_dataset, return_dataset_test
import torch.nn.functional as F
import metric.mmd as mmd
from perturb import PerturbationGenerator
from utils.utils import weights_init, group_step

# Training settings
parser = argparse.ArgumentParser(description='SSDA Classification')
parser.add_argument('--steps', type=int, default=50000, metavar='N',
                    help='maximum number of iterations '
                         'to train (default: 50000)')
parser.add_argument('--multi', type=float, default=0.1, metavar='MLT',
                    help='learning rate multiplication')
parser.add_argument('--T', type=float, default=0.05, metavar='T',
                    help='temperature (default: 0.05)')
parser.add_argument('--checkpath', type=str, default='./checkpoints',
                    help='dir to save checkpoint')
parser.add_argument('--net', type=str, default='resnet34',
                    help='which network to use')
parser.add_argument('--source', type=str, default='real',
                    help='source domain')
parser.add_argument('--target', type=str, default='sketch',
                    help='target domain')
parser.add_argument('--dataset', type=str, default='multi',
                    choices=['multi', 'office', 'office_home'],
                    help='the name of dataset')
parser.add_argument('--num', type=int, default=3,
                    help='number of labeled examples in the target')

args = parser.parse_args()
target_loader_test, class_list = return_dataset_test(args)
use_gpu = torch.cuda.is_available()

if args.net == 'resnet34':
    G = resnet34()
    inc = 512
elif args.net == "alexnet":
    G = AlexNetBase()
    inc = 4096
else:
    raise ValueError('Model cannot be recognized.')


if "resnet" in args.net:
    F1 = Predictor_deep_latent(num_class=len(class_list), inc=inc)
else:
    F1 = Predictor_latent(num_class=len(class_list), inc=inc, temp=args.T)
G = torch.nn.DataParallel(G).cuda()
F1 = torch.nn.DataParallel(F1).cuda()

G_dict = os.path.join(args.checkpath, "G_{}_{}_to_{}_step_{}.pth.tar".format(args.dataset, args.source, args.target, args.steps))
pretrained_dict = torch.load(G_dict)
model_dict = G.state_dict()
model_dict.update(pretrained_dict)
G.load_state_dict(model_dict)

F_dict = os.path.join(args.checkpath, "F1_{}_{}_to_{}_step_{}.pth.tar".format(args.dataset, args.source, args.target, args.steps))
pretrained_dict = torch.load(F_dict)
model_dict = F1.state_dict()
model_dict.update(pretrained_dict)
F1.load_state_dict(model_dict)



def test(base, classifier, loader):
    base.eval()
    classifier.eval()
    test_loss = 0
    correct = 0
    size = 0
    num_class = len(class_list)
    criterion = nn.CrossEntropyLoss().cuda()
    confusion_matrix = torch.zeros(num_class, num_class)
    with torch.no_grad():
        for batch_idx, data_t in enumerate(loader):
            im_data_t = Variable(data_t[0].cuda())
            gt_labels_t = Variable(data_t[1].cuda())
            feat = base(im_data_t)
            _, output1 = classifier(feat)
            size += im_data_t.size(0)
            pred1 = output1.data.max(1)[1]
            for t, p in zip(gt_labels_t.view(-1), pred1.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            correct += pred1.eq(gt_labels_t.data).cpu().sum()
            test_loss += criterion(output1, gt_labels_t) / len(loader)
    print('Test set: Average loss: {:.4f}, '
          'Accuracy: {}/{} F1 ({:.0f}%)'.
          format(test_loss, correct, size,
                 100. * correct / size))
    return test_loss.data, 100. * float(correct) / size

loss_test, acc_test = test(G, F1, target_loader_test)


