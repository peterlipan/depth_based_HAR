import os
import time
import shutil
import torch
import numpy as np
import torch.nn.functional as F
from .metrics import accuracy, validate, AverageMeter, compute_avg_metrics
from utils import HogRegressionLoss


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr_steps = args.lr_steps
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


def save_checkpoint(state, epoch, is_best, args):
    filename = os.path.join(args.checkpoints, 'epoch_{:d}_.pth'.format(epoch + 1))
    torch.save(state, filename)
    if is_best:
        best_name = os.path.join(args.checkpoints, 'best_{:s}_{:s}.pth'.format(args.dataset, args.modality))
        shutil.copyfile(filename, best_name)


def train(loaders, model, criterion, optimizer, scheduler, logger, args):
    losses = AverageMeter()
    hog_losses = AverageMeter()
    cls_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    # HOG loss function
    hog_criterion = HogRegressionLoss()

    # switch to train mode
    model.train()
    train_loader, test_loader = loaders
    cur_iter = 0
    iters = len(train_loader)
    best_top1 = 0
    start = time.time()
    for epoch in range(args.epochs):
        for i, (img, hog_features, target) in enumerate(train_loader):
            data_time.update(time.time() - start)
            img, hog_features, target = img.cuda(), hog_features.cuda().float(), target.cuda()

            output, hog_predictions = model(img)
            hog_loss = hog_criterion(hog_predictions, hog_features)
            cls_loss = criterion(output, target)
            loss = args.hog_weight * hog_loss + cls_loss

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), img.size(0))
            hog_losses.update(hog_loss.item(), img.size(0))
            cls_losses.update(cls_loss.item(), img.size(0))
            top1.update(prec1.item(), img.size(0))
            top5.update(prec5.item(), img.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - start)
            start = time.time()

            cur_iter += 1
            if cur_iter % 150 == 0:
                cur_lr = optimizer.param_groups[-1]["lr"]
                print(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                       'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                       'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                       'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, top1=top1, top5=top5, lr=cur_lr)))
                test_acc, test_f1, test_auc, test_bac, test_sens, test_spec, test_prec, test_loss = validate(
                    test_loader, model, criterion)
                if logger is not None:
                    logger.log({'Training': {'loss': losses.val,
                                             'Top-1 Accuracy': top1.val,
                                             'HOG loss': hog_losses.val,
                                             'Classification loss': cls_losses.val,
                                             'Top-5 Accuracy': top5.val,
                                             'Learning rate': cur_lr}})
                    logger.log({'Test': {'loss': test_loss,
                                         'Accuracy': test_acc,
                                         'F1 score': test_f1,
                                         'AUC': test_auc,
                                         'BAC': test_bac,
                                         'Sensitivity': test_sens,
                                         'Specificity': test_spec,
                                         "Precision": test_prec}})
        # update the learning rate
        scheduler.step()
        test_acc, test_f1, test_auc, test_bac, test_sens, test_spec, test_prec, test_loss = validate(test_loader, model,
                                                                                                     criterion)
        if logger is not None:
            logger.log({'Training': {'loss': losses.val,
                                     'Top-1 Accuracy': top1.val,
                                     'Top-5 Accuracy': top5.val}})
            logger.log({'Test': {'loss': test_loss,
                                 'Accuracy': test_acc,
                                 'F1 score': test_f1,
                                 'AUC': test_auc,
                                 'BAC': test_bac,
                                 'Sensitivity': test_sens,
                                 'Specificity': test_spec,
                                 "Precision": test_prec}})
        is_best = test_acc > best_top1
        best_top1 = max(best_top1, test_acc)
        save_checkpoint(model.module.state_dict(), epoch, is_best, args)


def test_time_training(dataloader, model, criterion, optimizer, logger, args):
    top1 = AverageMeter()
    top5 = AverageMeter()
    hog_losses = AverageMeter()
    cls_losses = AverageMeter()

    all_logits = torch.Tensor().cuda()
    all_targets = torch.Tensor().cuda()

    # HOG loss function
    hog_criterion = HogRegressionLoss()

    for i, (img, hog_features, target) in enumerate(dataloader):
        # training
        model.train()
        img, hog_features, target = img.cuda(), hog_features.cuda().float(), target.cuda()
        for _ in range(args.steps_per_sample):
            _, hog_predictions = model(img)
            hog_loss = hog_criterion(hog_predictions, hog_features)
            hog_losses.update(hog_loss.item(), img.size(0))

            optimizer.zero_grad()
            hog_loss.backward()
            optimizer.step()

        # test
        with torch.no_grad():
            model.eval()
            output, _ = model(img)
            logits = F.softmax(output, dim=1)

            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            loss = criterion(output, target)

            cls_losses.update(loss.item(), img.size(0))
            top1.update(prec1.item(), img.size(0))
            top5.update(prec5.item(), img.size(0))
            all_logits = torch.cat((all_logits, logits))
            all_targets = torch.cat((all_targets, target))

            print(('Image: [{0}/{1}]\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'Prec@1 {top1.avg:.3f}\t'
                   'Prec@5 {top5.avg:.3f}'.format(
                    i, len(dataloader), loss=cls_losses, top1=top1, top5=top5)))

        if logger is not None:
            logger.log({'TTT': {'Top-1 Accuracy': top1.avg,
                                     'HOG loss': hog_losses.avg,
                                     'Classification loss': cls_losses.avg,
                                     'Top-5 Accuracy': top5.avg}})

    acc, f1, auc, bac, sens, spec, prec = compute_avg_metrics(all_targets, all_logits)
    if logger is not None:
        logger.log({'Overall': {'Accuracy': acc,
                                 'F1 score': f1,
                                 'AUC': auc,
                                 'BAC': bac,
                                 'Sensitivity': sens,
                                 'Specificity': spec,
                                 "Precision": prec}})
