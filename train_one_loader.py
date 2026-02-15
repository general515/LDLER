import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from data_one import get_dataloaders
from model_new import create_model

from utils import set_random_seed, Logger, AverageMeter, get_accuracy, save_checkpoint

parser = argparse.ArgumentParser(description='PyTorch Training')
# train configs
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--lr', default=0.0002, type=float)
parser.add_argument('--gamma', default=0.98, type=float)
parser.add_argument('--num_classes', default=7, type=int)
# method configs
parser.add_argument('--threshold', default=0.7, type=float)
parser.add_argument('--alpha', default=None, type=float)
parser.add_argument('--beta', default=7, type=int)
parser.add_argument('--min_weight', default=0.2, type=float)
parser.add_argument('--drop_rate', default=0.0, type=float)

# common configs
parser.add_argument('--seed', default=None, type=int)
parser.add_argument('--dataset', default='raf', type=str)
parser.add_argument('--data_path', default='E:/BaiduWangpan/RAF/basic', type=str)
#parser.add_argument('--dataset', default='ferplus', type=str)
#parser.add_argument('--data_path', default=r'D:/datasets/fer2013_plus_combine', type=str)
#parser.add_argument('--dataset', default='affectnet', type=str)
#parser.add_argument('--data_path', default=r'E:\datasets\AffectNet\processed', type=str)
parser.add_argument('--num_workers', default=16, type=int)
parser.add_argument('--device_id', default=0, type=int)

args = parser.parse_args()

best_acc = 0
best_epoch = 0

# set device
device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')

# set random seed
if args.seed is not None:
    set_random_seed(args.seed)
def main():
    global best_acc
    global best_epoch
    global device
    # log file
    logger = Logger('./results/log-' + time.strftime('%b%d_%H-%M-%S') + '.txt')
    logger.info(args)

    # TensorBoard writer
    writer = SummaryWriter()

    # model
    logger.info('Load model...')
    model = create_model(args.num_classes, args.drop_rate).to(device)

    # dataloaders
    train_loader, test_loader = get_dataloaders(args.dataset, args.data_path, \
                                                args.batch_size, args.num_workers)

    # loss & optimizer.
    criterion = nn.CrossEntropyLoss(reduction='none')
    criterion_kld = nn.KLDivLoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    logger.info('Start training.')

    for epoch in range(1, args.epochs + 1):
        logger.info('----------------------------------------------------------')
        logger.info('Epoch: %d, Learning Rate: %f', epoch, optimizer.param_groups[0]['lr'])

        # train
        train_loss_ce, train_loss_kld = \
            train(train_loader, model, criterion, criterion_kld, optimizer, epoch)

        # test
        test_loss, test_acc = validate(test_loader, model, criterion, epoch)

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
        logger.info('')
        logger.info('Train SLL loss: %.4f', train_loss_ce)
        logger.info('Train LDL loss: %.4f', train_loss_kld)
        logger.info('Test Acc: %.2f', test_acc)
        logger.info('Test loss: %.4f', test_loss)
        logger.info('Best Acc: %.2f (%d)', best_acc, best_epoch)

        is_best = (best_epoch == epoch)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'acc': test_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict()
        },
            is_best)
        scheduler.step()

def train(train_loader, model, criterion, criterion_kld, optimizer, epoch):
    if args.alpha is not None:
        alpha_1 = args.alpha
        alpha_2 = 1 - args.alpha
    else:
        alpha_1 = (1 - 1 / (1 + np.exp(-(epoch - 1) / args.beta))) / 0.5
        alpha_2 = 1 / (1 + np.exp(-(epoch - 1) / args.beta))
    print('Alpha_1, Alpha_2: {:.4f}, {:.4f} Beta: {:d}'.format(alpha_1, alpha_2, args.beta))
    # losses
    losses_ce = AverageMeter()
    losses_kld = AverageMeter()

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    # training
    model.train()
    pbar.set_description(f'Epoch [{epoch}/{args.epochs}] training SLL branch')
    # training branch 1 (single label branch)
    for i, (images, labels, _) in pbar:
        images = images.to(device)
        labels = labels.to(device)
        outputs_1, _ = model(images, True)
        loss_ce = alpha_1 * criterion(outputs_1, labels).mean()
        # record loss
        losses_ce.update(loss_ce.item(), images.size(0))
        optimizer.zero_grad()
        loss_ce.backward()
        optimizer.step()
        pbar.set_postfix(loss=losses_ce.avg)

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    pbar.set_description(f'Epoch [{epoch}/{args.epochs}] Geting features')

    #features = torch.zeros(train_loader.sampler.num_samples, 512).to(device)
    probs = torch.zeros(train_loader.sampler.num_samples, args.num_classes).to(device)
    if args.dataset == 'ferplus':
        targets = torch.tensor(train_loader.dataset.targets).long().to(device)
    else:
        targets = torch.tensor(train_loader.dataset.label).long().to(device)

    model.eval()
    for i, (images, labels, idx) in pbar:
        images = images.to(device)
        with torch.no_grad():
            prob, _ = model(images, True)
        #features[idx] = feat
        probs[idx] = torch.softmax(prob, dim=1)

    LD = torch.zeros(args.num_classes, args.num_classes).to(device)
    for i in range(args.num_classes):
        tmp = probs[targets == i].mean(dim=0)
        if tmp[i] < args.threshold:
            LD[i] = torch.zeros(args.num_classes).fill_((1 - args.threshold) / \
                (args.num_classes - 1)).scatter_(0, torch.tensor(i), args.threshold)
        else:
            LD[i] = tmp

    entropy = torch.distributions.Categorical(probs=probs, validate_args=False).entropy()

    #Relabeling
    newtargets = targets.clone()
    for ind in range(probs.size(0)):
        maxind = torch.argmax(probs[ind])
        if maxind != targets[ind] and probs[ind][maxind] > 0.5:
            newtargets[ind] = maxind

    model.train()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    pbar.set_description(f'Epoch [{epoch}/{args.epochs}] Training LDL')
    for i, (images, labels, idx) in pbar:
        images = images.to(device)
        outputs_2, _ = model(images, False)

        weights = entropy[idx]
        weights = (weights - weights.min()) / (weights.max() - weights.min())
        weights = (1 - weights) * (1 - args.min_weight) + args.min_weight

        weights = weights.unsqueeze(1)
        ldl_target = LD[newtargets[idx]] * weights + probs[idx] * (1 - weights)
        loss_kld = alpha_2 * criterion_kld(F.log_softmax(outputs_2, dim=1), ldl_target).sum() / images.size(0)
        losses_kld.update(loss_kld.item(), images.size(0))
        optimizer.zero_grad()
        loss_kld.backward()
        optimizer.step()
        pbar.set_postfix(losses=losses_kld.avg)
    return losses_ce.avg, losses_kld.avg


def validate(test_loader, model, criterion, epoch):
    losses = AverageMeter()
    accs = AverageMeter()

    model.eval()

    outputs_new = torch.ones(1, args.num_classes).to(device)
    targets_new = torch.ones(1).long().to(device)

    pbar = tqdm(enumerate(test_loader), total=len(test_loader))
    pbar.set_description(f'Epoch [{epoch}/{args.epochs}] Testing')

    with torch.no_grad():
        for i, (inputs, targets, indexes) in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = model(inputs, False)
            loss = criterion(outputs, targets).mean()

            outputs_new = torch.cat((outputs_new, outputs), dim=0)
            targets_new = torch.cat((targets_new, targets), dim=0)
            top1, _ = get_accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            accs.update(top1.item(), inputs.size(0))

            pbar.set_postfix(loss=losses.avg, acc=accs.avg)

    return (losses.avg, accs.avg)


if __name__ == '__main__':
    main()