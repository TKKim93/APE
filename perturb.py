import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import contextlib


def set_requires_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad

@contextlib.contextmanager
def disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)

def l2_normalize(d):
    d_reshaped = d.view(d.size(0), -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d

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


class PerturbationGenerator(nn.Module):

    def __init__(self, feature_extractor, classifier, xi=1e-6, eps=3.5, ip=1):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.xi = xi
        self.eps = eps
        self.ip = ip
        self.kl_div = KLDivLossWithLogits()

    def forward(self, inputs):
        with disable_tracking_bn_stats(self.feature_extractor):
            with disable_tracking_bn_stats(self.classifier):
                features = self.feature_extractor(inputs)
                logits = self.classifier(features)[1].detach()

                # prepare random unit tensor
                d = l2_normalize(torch.randn_like(inputs).to(inputs.device))

                # calc adversarial direction
                x_hat = inputs
                x_hat = x_hat + self.xi * d
                x_hat.requires_grad = True
                features_hat = self.feature_extractor(x_hat)
                logits_hat = self.classifier(features_hat, reverse=True, eta=1)[1]
                prob_hat = F.softmax(logits_hat, 1)
                adv_distance = (prob_hat * torch.log(1e-4 + prob_hat)).sum(1).mean()
                adv_distance.backward()
                d = l2_normalize(x_hat.grad)
                self.feature_extractor.zero_grad()
                self.classifier.zero_grad()
                r_adv = d * self.eps
                return r_adv.detach(), features
