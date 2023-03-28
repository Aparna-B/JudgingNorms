import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import time
import copy
import os
from PIL import Image
from sklearn.metrics import mean_squared_error,roc_auc_score, confusion_matrix, classification_report, roc_curve, precision_recall_curve, auc
from scipy.stats import entropy
from scipy.spatial import distance
from math import log2
from sklearn import metrics

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet18":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name =="resnet50":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "resnet50_all":
        """ ResNET50 any violation prediction
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        # NB: num_classes is hardcoded to 4 here!!
        model_ft.fc = nn.Linear(num_ftrs, 4)
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

def train_model_all(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0
    best_auc = 0.5
    best_f1 = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            predictions_val = []
            targets_val = []

            # Iterate over data.
            # NB: Last output is the name of image
            # NB: Order of labels are OR label, factual features 1-3
            for inputs, labels1, labels2, labels3, labels4, _ in dataloaders[phase]:
                inputs = inputs.to(device)
                labels1 = labels1.to(device)
                labels2 = labels2.to(device)
                labels3 = labels3.to(device)
                labels4 = labels4.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    
                    #outputs = torch.squeeze(outputs)
                    labels1 = labels1.float()
                    labels2 = labels2.float()
                    labels3 = labels3.float()
                    labels4 = labels4.float()
                    try:
                        assert outputs.shape[1]==4
                    except AssertionError:
                        shapes=output.shape
                        print(shapes)
                    loss = 0.25*criterion(outputs[:, 0], labels1)
                    loss += 0.25*criterion(outputs[:, 1], labels2)
                    loss += 0.25*criterion(outputs[:, 2], labels3)
                    loss += 0.25*criterion(outputs[:, 3], labels4)
                

                    preds = torch.round(torch.sigmoid(outputs[:, 0]))
                    predictions_val.extend(torch.sigmoid(outputs[:, 0]).data.cpu().numpy())
                    targets_val.extend(labels1.data.cpu().numpy())
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                # NB: Here, labels1 is the OR of factual features, i.e., our primary prediction
                # target.
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels1.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double(
            ) / len(dataloaders[phase].dataset)
            fpr, tpr, thresholds = metrics.roc_curve(targets_val, predictions_val, pos_label=1)
            epoch_auc = metrics.auc(fpr, tpr)
            epoch_f1 = metrics.f1_score(targets_val, np.round(predictions_val), average='macro')


            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_f1 > best_f1:
                best_acc = epoch_acc
                best_auc = epoch_auc
                best_f1 = epoch_f1
                best_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print(f'Best epoch: {best_epoch}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

def get_transforms():
    # Data augmentation and normalization for training
    # Just normalization for validation and test
    # Normalizing constants from ImageNet (pretraining data)
    # NB: Verified that changing augmentations don't cause results to vary.
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomPerspective(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform, test_transform



def test_model_all(model, dataloader, device, img_dir, results_dir=None, save_str=None):
    viol = 0
    nonViol = 0
    violLs = []
    probLs = []
    predLs = []
    labelLs = []
    imgnames = []
    cont_label = []
    corr = 0
    total = 0
    probl1 = []
    probl2 = []
    probl3 = []

    model.eval()
    # get predictions on test set
    with torch.no_grad():
        # load images 1 by 1 for prediction
        for i, (image, label, label2, label3, label4, fname) in enumerate(dataloader, 0):
            image = image.to(device)
            label = label.to(device)
            label = label.float()
            output = model(image)
            cont_lab = torch.sigmoid(output[:, 0])
            outputs = output[:, 0]
            prob = torch.sigmoid(outputs)
            pred = torch.round(prob)
            if pred == label:
                corr += 1
            if pred == 1:
                violLs.append(fname[0])
                viol += 1
            else:
                nonViol += 1
            total += 1
            probLs.append(prob.cpu().data.numpy())
            predLs.append(pred.cpu().data.numpy())
            labelLs.append(int(label.cpu().data.numpy()))
            cont_label.append(cont_lab.cpu().data.numpy())
            imgnames.append(fname)
            probl1.append(torch.sigmoid(output[:, 1]).cpu().data.numpy())
            probl2.append(torch.sigmoid(output[:, 2]).cpu().data.numpy())
            probl3.append(torch.sigmoid(output[:, 3]).cpu().data.numpy())

    # print results including auc score, and save image files with violation
    acc = corr / total
    print(f"Test Accuracy = {acc}")
    print(f"Number of violations = {viol}")
    print(f"Number of non violations = {nonViol}")
    try:
        auc_score = roc_auc_score(labelLs, probLs)
    except:
        auc_score = 0
    print(f"AUC Score = {auc_score}")
    cm = confusion_matrix(labelLs, predLs)
    print(f"Confusion Matrix: {cm}")
    cl = classification_report(labelLs, predLs)
    print(f"Classification Report: {cl}")


    return imgnames, np.array(predLs), np.array(probLs), np.array(labelLs), np.array(cont_label),np.array(probl1),np.array(probl2),np.array(probl3)

