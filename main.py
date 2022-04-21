from utils.arguments import parse_arguments
from utils.dataset import CustomDataSet
from utils.transforms import get_transforms

from models.resnet50 import CustomResNet50

import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import torch
import torchvision
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support

import colorama
from colorama import Fore, Back, Style
colorama.init()


def show_batch(dl):
    for images, _, info in dl:
        fig, ax = plt.subplots(figsize=(32, 32))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images[:32], nrow=8).squeeze().permute(1, 2, 0).clamp(0,1))
        plt.savefig("./img/mygraph.png")
        break

def train(args, model, train_loader, nclasses, optimizer, criterion, epoch):
    model.train()
    correct = 0
    train_losses, preds, labels = [], [], []
    confusion_matrix = torch.zeros(nclasses, nclasses)

    for batch_idx, (data, target, _) in enumerate(train_loader):
        data, target = data.cuda(), target.long().cuda()
        output = model(data)
        optimizer.zero_grad()

        # calculate loss
        loss = criterion(output, target)
        train_losses.append(loss.item())

        # backward
        loss.backward()

        # optimizer step
        optimizer.step()

        # calculate correct predictions
        pred = F.softmax(output, dim=1).max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()

        # to compute metrics
        preds.append(pred.view(-1).cpu())
        labels.append(target.view(-1).cpu())
        
        for t, p in zip(target.view(-1), pred.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1

        # print log
        if batch_idx % args.log_interval == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tCELoss: {:.6f}'.format(epoch,
                                                                                        batch_idx * len(data),
            len(train_loader.dataset), 100. * batch_idx / len(train_loader),
            loss.item()))
    
    train_loss = np.mean(np.asarray(train_losses))
    # compute the metrics
    precision, recall, fscore, _ = precision_recall_fscore_support(
        y_true=torch.cat(labels), y_pred=torch.cat(preds), average='micro')

    # print epoch logs
    per_class_accuracy = confusion_matrix.diag() / confusion_matrix.sum(1)
    print(Fore.GREEN + '\nTrain set: Accuracy: {}/{}({:.2f}%), Average Loss: {:.4f}'.format(correct,
    len(train_loader.dataset), 100 * correct / len(train_loader.dataset), train_loss) +
    Style.RESET_ALL)

    print(Fore.GREEN + 'Classwise Accuracy:: Cl-0: {}/{}({:.2f}%),\
    Cl-1: {}/{}({:.2f}%), Cl-2: {}/{}({:.2f}%), Cl-3: {}/{}({:.2f}%); \
    Precision: {:.3f}, Recall: {:.3f}, F1: {:.3f}'.format(
    int(confusion_matrix.diag()[0].item()), int(confusion_matrix.sum(1)[0].item()), per_class_accuracy[0].item() * 100.,
    int(confusion_matrix.diag()[1].item()), int(confusion_matrix.sum(1)[1].item()), per_class_accuracy[1].item() * 100.,
    int(confusion_matrix.diag()[2].item()), int(confusion_matrix.sum(1)[2].item()), per_class_accuracy[2].item() * 100.,
    int(confusion_matrix.diag()[3].item()), int(confusion_matrix.sum(1)[3].item()), per_class_accuracy[3].item() * 100.,
    precision, recall, fscore) + Style.RESET_ALL)

    return model

    

def test(args, model, test_loader, nclasses, criterion, epoch, state_dict, weights_path):
    model.eval()
    correct = 0
    test_losses, preds, labels = [], [], []
    confusion_matrix = torch.zeros(nclasses, nclasses)
    
    with torch.no_grad():
        for data, target, _ in test_loader:
            data, target = data.cuda(), target.long().cuda()
            output = model(data)

            # calculate loss
            loss = criterion(output, target)
            test_losses.append(loss.item())

            # calculate correct predictions
            pred = F.softmax(output, dim=1).max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

            # to compute metrics
            preds.append(pred.view(-1).cpu())
            labels.append(target.view(-1).cpu())
            
            for t, p in zip(target.view(-1), pred.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    
    test_loss = np.mean(np.asarray(test_losses))

    precision, recall, fscore, _ = precision_recall_fscore_support(
        y_true=torch.cat(labels), y_pred=torch.cat(preds), average='micro')
    per_class_accuracy = confusion_matrix.diag() / confusion_matrix.sum(1)
    print(Fore.RED + '\nTest Set: Average Loss: {:.4f}, Accuracy: {}/{} \
    ({:.2f}%)'.format(test_loss, correct, len(test_loader.dataset), 100 *
    correct / len(test_loader.dataset))  + Style.RESET_ALL)

    print(Fore.RED + 'Classwise Accuracy:: Cl-0: {}/{}({:.2f}%), Cl-1: {}/{}({:.2f}%) \
    Cl-2: {}/{}({:.2f}%), Cl-3: {}/{}({:.2f}%); \
    Precision: {:.3f}, Recall: {:.3f}, F1: {:.3f}'.format(
    int(confusion_matrix.diag()[0].item()), int(confusion_matrix.sum(1)[0].item()), per_class_accuracy[0].item() * 100.,
    int(confusion_matrix.diag()[1].item()), int(confusion_matrix.sum(1)[1].item()), per_class_accuracy[1].item() * 100.,
    int(confusion_matrix.diag()[2].item()), int(confusion_matrix.sum(1)[2].item()), per_class_accuracy[2].item() * 100.,
    int(confusion_matrix.diag()[3].item()), int(confusion_matrix.sum(1)[3].item()), per_class_accuracy[3].item() * 100.,
    precision, recall, fscore) + Style.RESET_ALL)

    metrics = {'test/accuracy': correct / len(test_loader.dataset) * 100.,
              'test/precision': precision,
              'test/recall': recall,
              'test/F1': fscore,
              'test/loss': test_loss}
    
    print('Saving weights...')
    save_weights(model, os.path.join(weights_path, 'model.pth'))
    save_best_model(model, weights_path, metrics, state_dict)

def save_weights(model, path):
    torch.save(model.state_dict(), path)

def load_weights(args, model, path):
    if args.arch == 'ResNet50':
        state_dict_ = torch.load(path)
        modified_state_dict = {}
        for key in state_dict_.keys():
            mod_key = key[7:]
            modified_state_dict.update({mod_key: state_dict_[key]})
    else:
        modified_state_dict = torch.load(path)
    model.load_state_dict(modified_state_dict, strict=True)
    return model

def save_best_model(model, path, metrics, state_dict):
    if metrics['test/F1'] > state_dict['best_f1']:
        state_dict['best_f1'] = max(metrics['test/F1'], state_dict['best_f1'])
        state_dict['accuracy'] = metrics['test/accuracy']
        state_dict['precision'] = metrics['test/precision']
        state_dict['recall'] = metrics['test/recall']
        print('F1 score improved over the previous. Saving model...')
        save_weights(model=model, path=os.path.join(path, 'best_model.pth'))
    best_str = "Best Metrics:" + '; '.join(["%s - %s" % (k, v) for k, v in state_dict.items()])
    print(Fore.BLUE + best_str + Style.RESET_ALL)

def experiments(args):
    # load data
    train_data = pd.read_csv(os.path.join(args.dataset_root, 'train_frames.csv'))
    test_data = pd.read_csv(os.path.join(args.dataset_root, 'test_frames.csv'))
    
    # subset the dataset
    train_ds = CustomDataSet(args, train_data, transforms=get_transforms(args, 'train'))
    test_ds = CustomDataSet(args, test_data, transforms=get_transforms(args, 'test'))

    train_labels = train_data['frame_score'].unique()
    num_calss = len(train_labels)
    
    # data loader
    train_dloader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=True)
    test_dloader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False)

    # create log directories
    args.weights_dir = os.path.join('logs', args.run_name, 'weights')
    os.makedirs(args.weights_dir, exist_ok=True)

    # model
    model = CustomResNet50(args.img_size, args.img_size, num_calss, args.dropout)
    print('Number of params in the model: {}'.format(
        *[sum([p.data.nelement() for p in net.parameters()]) for net in [model]]))
    model = model.cuda()

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[70], gamma=0.1) # 10, 50
    
    # loss function
    criterion = nn.CrossEntropyLoss()

    state_dict = {'best_f1': 0., 'precision': 0., 'recall': 0., 'accuracy': 0.}

    for epoch in range(args.epochs):
        model = train(args, model, train_dloader, len(list(set(train_labels))), optimizer, criterion, epoch)
        test(args, model, test_dloader, len(list(set(train_labels))), criterion, epoch, state_dict, args.weights_dir)
        exp_lr_scheduler.step()

if __name__ == '__main__':
    args = parse_arguments()
    print(args)
    experiments(args)