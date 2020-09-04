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
from utils.return_dataset import return_dataset
import torch.nn.functional as F
import metric.mmd as mmd
from perturb import PerturbationGenerator
from utils.utils import weights_init, group_step

# Training settings
parser = argparse.ArgumentParser(description='SSDA Classification')
parser.add_argument('--steps', type=int, default=50000, metavar='N',
                    help='maximum number of iterations '
                         'to train (default: 50000)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--multi', type=float, default=0.1, metavar='MLT',
                    help='learning rate multiplication')
parser.add_argument('--T', type=float, default=0.05, metavar='T',
                    help='temperature (default: 0.05)')
parser.add_argument('--save_check', action='store_true', default=False,
                    help='save checkpoint or not')
parser.add_argument('--checkpath', type=str, default='./checkpoints',
                    help='dir to save checkpoint')
parser.add_argument('--save_interval', type=int, default=5000, metavar='N',
                    help='how many batches to wait before saving a model')
parser.add_argument('--net', type=str, default='alexnet',
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
parser.add_argument('--thr', type=float, default=0.5,
                    help='threshold for exploration scheme')

args = parser.parse_args()
source_loader, target_loader, target_loader_unl, target_loader_val, \
    target_loader_test, class_list = return_dataset(args)
use_gpu = torch.cuda.is_available()


if args.net == 'resnet34':
    G = resnet34()
    inc = 512
elif args.net == "alexnet":
    G = AlexNetBase()
    inc = 4096
else:
    raise ValueError('Model cannot be recognized.')

params = []
for key, value in dict(G.named_parameters()).items():
    if value.requires_grad:
        if 'classifier' not in key:
            params += [{'params': [value], 'lr': args.multi,
                        'weight_decay': 0.0005}]
        else:
            params += [{'params': [value], 'lr': args.multi * 10,
                        'weight_decay': 0.0005}]

if "resnet" in args.net:
    F1 = Predictor_deep_latent(num_class=len(class_list), inc=inc)
else:
    F1 = Predictor_latent(num_class=len(class_list), inc=inc, temp=args.T)
weights_init(F1)
lr = args.lr
G = torch.nn.DataParallel(G).cuda()
F1 = torch.nn.DataParallel(F1).cuda()


if os.path.exists(args.checkpath) == False:
    os.mkdir(args.checkpath)


def train():
    G.train()
    F1.train()
    optimizer_g = optim.SGD(params, momentum=0.9,
                            weight_decay=0.0005, nesterov=True)
    optimizer_f = optim.SGD(list(F1.parameters()),
                            lr=1.0, momentum=0.9, weight_decay=0.0005, nesterov=True)
    param_lr_g = []
    for param_group in optimizer_g.param_groups:
        param_lr_g.append(param_group["lr"])
    param_lr_f = []
    for param_group in optimizer_f.param_groups:
        param_lr_f.append(param_group["lr"])


    ################################################################################################################
    ################################################# train model ##################################################
    ################################################################################################################

    def zero_grad_all():
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()

    class AbstractConsistencyLoss(nn.Module):
        def __init__(self, reduction='mean'):
            super().__init__()
            self.reduction = reduction

        def forward(self, logits1, logits2):
            raise NotImplementedError

    class KLDivLossWithLogits(AbstractConsistencyLoss):
        def __init__(self, reduction='mean'):
            super().__init__(reduction)
            self.kl_div_loss = nn.KLDivLoss(reduction=reduction)

        def forward(self, logits1, logits2):
            return self.kl_div_loss(F.log_softmax(logits1, dim=1), F.softmax(logits2, dim=1))

    class EntropyLoss(nn.Module):
        def __init__(self, reduction='mean'):
            super().__init__()
            self.reduction = reduction

        def forward(self, logits):
            p = F.softmax(logits, dim=1)
            elementwise_entropy = -p * F.log_softmax(logits, dim=1)
            if self.reduction == 'none':
                return elementwise_entropy

            sum_entropy = torch.sum(elementwise_entropy, dim=1)
            if self.reduction == 'sum':
                return sum_entropy

            return torch.mean(sum_entropy)


    P = PerturbationGenerator(G, F1, xi=1, eps=25, ip=1)
    criterion = nn.CrossEntropyLoss().cuda()
    criterion_reduce = nn.CrossEntropyLoss(reduce=False).cuda()
    target_consistency_criterion = KLDivLossWithLogits(reduction='mean').cuda()
    criterion_entropy = EntropyLoss()

    all_step = args.steps
    data_iter_s = iter(source_loader)
    data_iter_t = iter(target_loader)
    data_iter_t_unl = iter(target_loader_unl)
    len_train_source = len(source_loader)
    len_train_target = len(target_loader)
    len_train_target_semi = len(target_loader_unl)
    best_acc = 0
    counter = 0
    if args.net == 'resnet34':
        thr = 0.5
    else:
        thr = 0.3  
  
    for step in range(all_step):
        optimizer_g = inv_lr_scheduler(param_lr_g, optimizer_g, step,
                                       init_lr=args.lr)
        optimizer_f = inv_lr_scheduler(param_lr_f, optimizer_f, step,
                                       init_lr=args.lr)
        lr = optimizer_f.param_groups[0]['lr']

        if step % len_train_target == 0:
            data_iter_t = iter(target_loader)
        if step % len_train_target_semi == 0:
            data_iter_t_unl = iter(target_loader_unl)
        if step % len_train_source == 0:
            data_iter_s = iter(source_loader)


        data_t = next(data_iter_t)
        data_t_unl = next(data_iter_t_unl)
        data_s = next(data_iter_s)

        im_data_s = Variable(data_s[0].cuda())
        gt_labels_s = Variable(data_s[1].cuda())
        im_data_t = Variable(data_t[0].cuda())
        gt_labels_t = Variable(data_t[1].cuda())
        im_data_tu = Variable(data_t_unl[0].cuda())
        gt_labels_tu = Variable(data_t_unl[1].cuda())
        gt_labels = torch.cat((gt_labels_s, gt_labels_t), 0)
        gt_dom_s = Variable(torch.zeros(im_data_s.size(0)).cuda().long())
        gt_dom_t = Variable(torch.ones(im_data_t.size(0)).cuda().long())
        gt_dom = torch.cat((gt_dom_s, gt_dom_t))
        zero_grad_all()

        ################################################################################################################
        ################################################# train model ##################################################
        ################################################################################################################
        data = torch.cat((im_data_s, im_data_t), 0)
        target = torch.cat((gt_labels_s, gt_labels_t), 0)
        sigma = [1, 2, 5, 10]

        output = G(data)
        output_tu = G(im_data_tu)
        latent_F1, out1 = F1(output)
        latent_F1_tu, out_F1_tu = F1(output_tu)

        # supervision loss
        loss = criterion(out1, target)

        # attraction scheme
        loss_msda = 10 * mmd.mix_rbf_mmd2(latent_F1, latent_F1_tu, sigma)

        # exploration scheme
        pred = out_F1_tu.data.max(1)[1].detach()
        ent = - torch.sum(F.softmax(out_F1_tu, 1) * (torch.log(F.softmax(out_F1_tu, 1) + 1e-5)), 1)
        mask_reliable = (ent < thr).float().detach()
        loss_cls_F1 = (mask_reliable * criterion_reduce(out_F1_tu, pred)).sum(0) / (1e-5 + mask_reliable.sum())

        (loss + loss_cls_F1 + loss_msda).backward(retain_graph=False)
        group_step([optimizer_g, optimizer_f])
        zero_grad_all()
        if step % 20 == 0:
            print('step %d' % step, 'loss_cls: {:.4f}'.format(loss.cpu().data), ' | ', 'loss_Attract: {:.4f}'.format(loss_msda.cpu().data), ' | ', \
                  'loss_Explore: {:.4f}'.format(loss_cls_F1.cpu().item()), end=' | ')

        # perturbation scheme
        bs = gt_labels_s.size(0)
        target_data = torch.cat((im_data_t, im_data_tu), 0)
        perturb, clean_vat_logits = P(target_data)
        perturb_inputs = target_data + perturb
        perturb_inputs = torch.cat(perturb_inputs.split(bs), 0)
        perturb_features = G(perturb_inputs)
        perturb_logits = F1(perturb_features)[0]
        target_vat_loss2 = 10 * target_consistency_criterion(perturb_logits, clean_vat_logits)

        target_vat_loss2.backward()
        group_step([optimizer_g, optimizer_f])
        zero_grad_all()


        if step % 20 == 0:
            print('loss_Perturb: {:.4f}'.format(target_vat_loss2.cpu().data))
        G.zero_grad()
        F1.zero_grad()
        zero_grad_all()

        if step % args.save_interval == 0 and step > 0:
            loss_test, acc_test = test(G, F1, target_loader_test)
            loss_val, acc_val = test(G, F1, target_loader_val)
            G.train()
            F1.train()

            if args.save_check:
                print('saving model')
                torch.save(G.state_dict(),
                           os.path.join(args.checkpath,
                                        "G_iter_model_{}_{}_"
                                        "to_{}_step_{}.pth.tar".
                                        format(args.method, args.source,
                                               args.target, step)))
                torch.save(F1.state_dict(),
                           os.path.join(args.checkpath,
                                        "F1_iter_model_{}_{}_"
                                        "to_{}_step_{}.pth.tar".
                                        format(args.method, args.source,
                                               args.target, step)))


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


train()


