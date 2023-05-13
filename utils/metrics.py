import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score, precision_score
from imblearn.metrics import sensitivity_score, specificity_score


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        output = F.softmax(output, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def compute_avg_metrics(groundTruth, logits):
    groundTruth = groundTruth.cpu().detach().numpy()
    logits = logits.cpu().detach().numpy()
    predictions = np.argmax(logits, -1)
    mean_acc = accuracy_score(y_true=groundTruth, y_pred=predictions)
    f1_macro = f1_score(y_true=groundTruth, y_pred=predictions, average='macro')
    try:
        auc = roc_auc_score(y_true=groundTruth, y_score=logits, multi_class='ovr')
    except ValueError as error:
        print('Error in computing AUC. Error msg:{}'.format(error))
        auc = 0
    bac = balanced_accuracy_score(y_true=groundTruth, y_pred=predictions)
    sens_macro = sensitivity_score(y_true=groundTruth, y_pred=predictions, average='macro')
    spec_macro = specificity_score(y_true=groundTruth, y_pred=predictions, average='macro')
    prec_macro = precision_score(y_true=groundTruth, y_pred=predictions, average="macro")

    return mean_acc, f1_macro, auc, bac, sens_macro, spec_macro, prec_macro


def validate(loader, model, criterion):
    training = model.training
    # switch to evaluate mode
    model.eval()

    losses = AverageMeter()
    groundTruth = torch.Tensor().cuda()
    logits = torch.Tensor().cuda()

    with torch.no_grad():
        for i, (img, target) in enumerate(loader):
            img, target = img.cuda(), target.cuda()

            # compute output
            output = model(img)
            loss = criterion(output, target)
            output = F.softmax(output, dim=1)

            losses.update(loss.item(), img.size(0))
            groundTruth = torch.cat((groundTruth, target))
            logits = torch.cat((logits, output))

        acc, f1, auc, bac, sens, spec, prec = compute_avg_metrics(groundTruth, logits)

    model.train(training)

    return acc, f1, auc, bac, sens, spec, prec, losses.avg
