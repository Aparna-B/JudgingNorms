import modelling
import os
from pathlib import Path
import pandas as pd
import torch
from ImageDataset import ImageDatasetAll
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import copy
import random
from contention_data_split import get_data_split, get_external_data_split

class Classifier:

    def __init__(self, opts):
        self.data_root = opts.data_root
        self.img_root = opts.img_root
        self.experiment = opts.experiment
        self.load_path = opts.load_path
        self.category = opts.category
        self.csv_file = opts.csv_file
        self.num_images = opts.num_images
        self.labels_per_image = opts.labels_per_image
        self.input_size = opts.input_size
        self.batch_size = opts.batch_size
        self.learning_rate = opts.learning_rate
        self.learning_rate_str = str(opts.learning_rate).replace('.', '_')
        self.ngpu = opts.ngpu
        self.model_name = opts.model_name
        self.num_epochs = opts.num_epochs
        self.num_workers = opts.num_workers
        self.feature_extract = opts.feature_extract
        self.split = opts.split
        self.seed = opts.seed
        self.thresholds = opts.thresholds
        self.cross = opts.cross
        self.transfer = opts.transfer
        self.transfer_path = opts.transfer_path
        self.test_only = opts.test_only
        self.contention_ref = opts.contention_ref
        self.contention = opts.contention
        self.external_dataset = opts.external_dataset
        self.train_size_red = opts.train_size_red
        self.label_noise = opts.label_noise
        self.dataset_name = opts.dataset_name
        self.momentum = opts.momentum
        self.weight_decay = opts.weight_decay


    def csv_split_contention(self, cross=-1):
        '''
        splits csv file into training validation and test split based on input parameters
        :return:
        '''
        csv_path = os.path.join(self.data_root, self.csv_file.format(self.experiment))
        df_raw = pd.read_csv(csv_path)
        # Extract img names and label column for this run
        try:
            for exp in ['descriptive', 'normative']:
                df_raw[exp + '0'] = (df_raw[exp + '1'] |
                                     df_raw[exp + '2'] | df_raw[exp + '3']).astype(int)
        except:
            print('Check data')

        if (cross ==0) :
            np.random.seed(1)
            assert self.contention_ref=='normative'
            n_num=0
            train_imgs, val_imgs, test_imgs = get_data_split(self.data_root, self.csv_file.format(self.contention_ref),
                                                            self.num_images,
                                                             self.labels_per_image, cont_cat=self.contention_ref, p=self.contention,
                                                             n_num=n_num)
            np.random.seed(self.seed)
        else:
            raise NotImplementedError

        
        # split data based on train, val, test - no need to shuffle since image
        # order random already
        total_num = self.num_images * self.labels_per_image
        np.random.seed(self.seed)


        df = df_raw[['imgname', self.experiment + '0',
                     self.experiment + '1', self.experiment + '2',
                     self.experiment +'3']]
        

        df = df[['imgname', self.experiment + '0',
                     self.experiment + '1', self.experiment + '2',
                     self.experiment +'3']]


        # for experiments with reduction in size
        new_train_size = int(self.train_size_red*len(train_imgs))
        new_val_size = int(self.train_size_red*len(val_imgs))
        # NB: Sampling without replacement
        train_imgs = random.sample(train_imgs, new_train_size)
        val_imgs = random.sample(val_imgs, new_val_size)
        assert len(val_imgs) == new_val_size

        df_train = df[df.imgname.isin(train_imgs)]
        df_val = df[df.imgname.isin(val_imgs)]
        df_test = df[df.imgname.isin(test_imgs)]

        train_y = df_train[[self.experiment + '0',
                     self.experiment + '1', self.experiment + '2',
                     self.experiment +'3']].values
        val_y = df_val[[self.experiment + '0',
                     self.experiment + '1', self.experiment + '2',
                     self.experiment +'3']].values

        if self.label_noise>0:
            train_y_ori=train_y.copy()
            val_y_ori=val_y.copy()
            sel_ind = np.random.choice(len(train_y), int(self.label_noise*len(train_y)),
                    replace=False)
            train_y[sel_ind] = 1- train_y[sel_ind]

            val_sel_ind = np.random.choice(len(val_y), int(self.label_noise*len(val_y)),
                    replace=False)
            val_y[val_sel_ind] = 1-val_y[val_sel_ind]

            df_train.loc[:,[self.experiment + '0',
                     self.experiment + '1', self.experiment + '2',
                     self.experiment +'3']] = train_y
            df_val.loc[:,[self.experiment + '0',
                     self.experiment + '1', self.experiment + '2',
                     self.experiment +'3']] = val_y
            for i in range(4):
                assert np.array_equal(df_train[self.experiment + str(i)].values,
                        train_y[:,i])
                assert np.array_equal(df_val[self.experiment + str(i)].values,
                        val_y[:,i])
                assert not np.array_equal(train_y[:,i], train_y_ori[:,i])
                assert not np.array_equal(val_y[:,i],val_y_ori[:,i])

        
        print('Test set shape is: ', df_test.shape)
        print('Intersection: ', len(set(df_train['imgname'].unique()).intersection(
            set(df_test['imgname'].unique()))))
        return df_train, df_val, df_test

    def run(self):
        '''
        Goes through entire training validation and test phase
        :return: Returns when job is complete
        '''

        # First create any directories needed for this run if not already
        # created and check for existence
        data_dir = os.path.join(self.data_root, self.img_root)
        if not os.path.exists(data_dir):
            print("Error: Data Directory Invalid. Please specify a valid directory.")
            exit()

        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        
        # Initialize the model for this run
        # By default, num_classes is 1. 
        # NB: For our case, it is hardcoded into model (its a fixed parameter).
        model_ft, input_size = modelling.initialize_model(
            self.model_name, 1, self.feature_extract, use_pretrained=True)

        # Print the model we just instantiated
        print(model_ft)

        # Data transform
        train_transform, val_transform, test_transform = modelling.get_transforms()

        print("Initializing Datasets and Dataloaders...")

        # Create training, validation datasets

        df_train, df_val, df_test_same = self.csv_split_contention(cross=0)
        img_dir = os.path.join(self.data_root, self.img_root)
        train_dataset = ImageDatasetAll(
            df=df_train, root_dir=img_dir, transform=train_transform)
        val_dataset = ImageDatasetAll(
            df=df_val, root_dir=img_dir, transform=val_transform)
        image_datasets = {'train': train_dataset, 'val': val_dataset}
        np.random.seed(self.seed)
        # Create training, validation dataloaders
        dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                           batch_size=self.batch_size,
                                                           shuffle=True,
                                                           num_workers=self.num_workers)
                            for x in ['train', 'val']}
        # Detect if we have a GPU available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Send the model to GPU
        model_ft = model_ft.to(device)

        # If transfer learning, load pretrained weights
        # NB: By default this path is an empty string.
        if self.transfer:
            model_ft.load_state_dict(torch.load(
                self.transfer_path, map_location=device))

        # Gather the parameters to be optimized/updated in this run. If we are
        # finetuning we will be updating all parameters. However, if we are
        # doing feature extract method, we will only update the parameters
        # that we have just initialized, i.e. the parameters with requires_grad
        # is True.
        params_to_update = model_ft.parameters()
        print("Params to learn:")
        if self.feature_extract:
            params_to_update = []
            for name, param in model_ft.named_parameters():
                if param.requires_grad:
                    params_to_update.append(param)
                    print("\t", name)
        else:
            for name, param in model_ft.named_parameters():
                if param.requires_grad:
                    print("\t", name)

        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(
            params_to_update, lr=self.learning_rate, momentum=self.momentum)

        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(
            optimizer_ft, step_size=7, gamma=self.weight_decay)

        # Setup the loss fxn
        criterion = nn.BCEWithLogitsLoss()

        # Train and evaluate
        model_ft, hist = modelling.train_model_all(model_ft, dataloaders_dict, criterion, optimizer_ft, exp_lr_scheduler,
                                                   device, num_epochs=self.num_epochs)

        """
        # save model weights
        save_str = f'c={self.category}-ex={self.experiment}-m={self.model_name}-s={self.seed}-' \
            f'ep={self.learning_rate_str}-b={self.batch_size}-f={self.feature_extract}-t={self.transfer}'
        torch.save(model_ft.state_dict(), os.path.join(models_dir, f'{save_str}.pt'))
        """

        test_dataset = ImageDatasetAll(
            df=df_val, root_dir=img_dir, transform=val_transform)
        testloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=1,
            shuffle=False, num_workers=self.num_workers)
        imgnames, pred, prob, labels, cont_label, prob1, prob2, prob3 = modelling.test_model_all(
            model_ft, testloader, device, img_dir, results_dir=None, save_str=None)
        if self.train_size_red !=-1 and self.label_noise!=-1:
            pd.DataFrame({'img': imgnames, 'cont':cont_label[:,0],'pred': pred[:,0], 'prob1':prob1[:,0],'prob2':prob2[:,0], 'prob3':prob3[:,0],'prob':prob[:,0],'labels': labels}).to_csv(
                os.path.join(f'./output_pred/{self.dataset_name}/hyperparam_tuning_f1',
                             f'contention_ref_{self.contention_ref}+model_{self.model_name}+batch_{self.batch_size}+lr_{self.learning_rate}+momentum_{self.momentum}+weightdecay_{self.weight_decay}+{self.contention}_contention+{self.experiment}_cat+{self.category}_seed+{self.seed}_validation_cross+0_size+{self.train_size_red}_noise+{self.label_noise}.csv'))
       

        # saving output appropriately
        if self.train_size_red !=1 or self.label_noise!=0:
            output_folder = f'{self.dataset_name}/noising_subsampling_f1'
        else:
            output_folder = f'{self.dataset_name}/test_predictions_f1'

        
        test_dataset = ImageDatasetAll(
                df=df_test_same, root_dir=img_dir, transform=test_transform)
        testloader = torch.utils.data.DataLoader(
                test_dataset, batch_size=1,
                shuffle=False, num_workers=self.num_workers)
        imgnames, pred, prob, labels, cont_label, prob1, prob2, prob3 = modelling.test_model_all(
                model_ft, testloader, device, img_dir, results_dir=None, save_str=None)
        pd.DataFrame({'img': imgnames, 'cont':cont_label[:,0],'pred': pred[:,0], 'prob1':prob1[:,0],'prob2':prob2[:,0], 'prob3':prob3[:,0],'prob':prob[:,0],'labels': labels}).to_csv(
                os.path.join(f'./output_pred/{output_folder}',
                             f'contention_ref_{self.contention_ref}+model_{self.model_name}+batch_{self.batch_size}_batch+lr_{self.learning_rate}+momentum_{self.momentum}+weightdecay_{self.weight_decay}+{self.contention}_contention+{self.experiment}_cat+{self.category}_seed+{self.seed}_cross+0_size+{self.train_size_red}_noise+{self.label_noise}.csv'))
        


