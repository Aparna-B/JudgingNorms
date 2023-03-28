import numpy as np
import random
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoConfig, AutoModel,AutoModelForSequenceClassification, AutoTokenizer, BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from transformers import BertPreTrainedModel, BertModel
import torch.nn as nn
from detoxify import Detoxify
import argparse
from focal_loss.focal_loss import FocalLoss
import torch.nn.functional as F
from text_data_split import get_data_split, get_external_data_split
# NB: Seed is changed in main functions.
np.random.seed(1)

# hyperparameters
# NB: These parameters are chosen based on Detoxify model.
lr = 3e-5
epochs = 10



class WeightedFocalLoss(nn.Module):
    "Weighted version of Focal Loss"

    def __init__(self, alpha=.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1 - alpha]).cuda()
        self.gamma = gamma

    def forward(self, inputs, targets):
        device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        w = torch.ones(1, 1).to(device)
        # print(targets)
        BCE_loss = F.binary_cross_entropy_with_logits(
            input=inputs, target=targets, pos_weight=torch.tensor(2).to(device), reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at * (1 - pt)**self.gamma * BCE_loss
        return F_loss.mean()

# NB: Calling it "Bert" base, but base model can be any AutoModel
class BertMultiSequenceClassification(nn.Module):

    def __init__(self, config, num_labels):
        super(BertMultiSequenceClassification, self).__init__()
        self.num_labels = num_labels
        self.config = AutoConfig.from_pretrained(config)
        self.bert = AutoModel.from_config(self.config)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, self.num_labels)

        self.bert.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class Dataset(torch.utils.data.Dataset):

    def __init__(self, encodings, labels, texts):
        self.encodings = encodings
        self.labels = labels
        self.texts = texts

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        item['texts'] = self.texts[idx]
        return item

    def __len__(self):
        return len(self.labels)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="contention-based text classifier.")
    parser.add_argument(
        "-p",
        "--contention",
        help="contention level",
        action="store",
        type=float,
        default=0.4,
        required=False)
    # NB: this is equivalent to the "experiment" flag in the image experiments
    # Used for choosing the training dataset condition.
    parser.add_argument(
        "-c",
        "--cross",
        help="descriptive=1; normative=0",
        action="store",
        type=int,
        default=0,
        required=False)
    parser.add_argument("--batch_size", help="batch size", action="store",
                        type=int, default=32, required=False)
    parser.add_argument("--seed", help="seed", action="store",
                        type=int, default=1, required=False)
    parser.add_argument("--weight", help="weight",
                        action="store", type=int, default=3, required=False)
    parser.add_argument("--model_name", help="model name", action="store", 
                type=str, default='bert-base-uncased', required=False)
    parser.add_argument("--train_size_red", help="proportion to reduce train set",
                        action="store", type=float, default=1, required=False)
    parser.add_argument("--label_noise_prop", help="label noise",
                        action="store", type=float, default=0, required=False)

    parser.add_argument("--contention_ref", help="reference for contention-aware train/dev/test split",
                        action="store", type=str, default='normative', required=False)

    args = parser.parse_args()

    np.random.seed(args.seed)
    model_name = args.model_name
    bs = args.batch_size
    print(args) 
    #NB: Training dataset selection
    experiment = 'normative'
    if args.cross == 1:
        experiment = 'descriptive'

    # Reading corresponding data
    df = pd.read_csv('{}_labels.csv'.format(experiment))
    all_texts = df['text'].unique()


    # this is splitting data wrt contention reference
    train_imgs, val_imgs, test_imgs = get_data_split(
        '{}_labels.csv'.format(args.contention_ref), args.contention_ref, args.contention)
    
    # NB: Note that in data splitting file the seed is set to 1 (for reproducibility of data splits).
    # Here, we set seeds for model training reproducibility.
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    
    # subsampling
    new_train_size = int(args.train_size_red*len(train_imgs))
    new_val_size = int(args.train_size_red*len(val_imgs))
    # NB: sampling without replacement
    train_imgs = random.sample(train_imgs, new_train_size)
    val_imgs = random.sample(val_imgs, new_val_size)
    assert len(train_imgs) == new_train_size
    
    df_train = df[df.imgname.isin(train_imgs)]
    df_val = df[df.imgname.isin(val_imgs)]
    df_test = df[df.imgname.isin(test_imgs)]
    train_texts = df_train['text'].values.tolist()
    val_texts = df_val['text'].values.tolist()
    if args.cross==0:
        train_y = df_train[
            ['normative1', 'normative2', 'normative3', 'normative0']].values
        val_y = df_val[
        ['normative1', 'normative2', 'normative3', 'normative0']].values
    else:
        train_y = df_train[['descriptive1', 'descriptive2',
         'descriptive3', 'descriptive0']].values
        val_y = df_val[
        ['descriptive1', 'descriptive2',
         'descriptive3', 'descriptive0']].values
        
    if args.label_noise_prop >0:
        sel_ind = np.random.choice(len(train_y), int(args.label_noise_prop*len(train_y)),
                replace=False)
        train_y[sel_ind]=1-train_y[sel_ind]
        # NB: Verified that modifications like this to noising procedure don't cause
        # results to vary much
        #train_y[sel_ind,:-1] = 1- train_y[sel_ind,:-1]
        #train_y[:,-1] = np.array(np.sum(train_y[:,:-1],axis=1)>0,dtype=np.int8)

        val_sel_ind = np.random.choice(len(val_y), int(args.label_noise_prop*len(val_y)),
                    replace=False)
        val_y[val_sel_ind]=1-val_y[val_sel_ind]
        #val_y[val_sel_ind,:-1] = 1-val_y[val_sel_ind,:-1]
        #val_y[:,-1] = np.array(np.sum(val_y[:,:-1],axis=1)>0,dtype=np.int8)

    
    # test set
    test_texts = df_test['text'].values.tolist()
    test_y = np.zeros((df_test.shape[0],4))
    test_cross_y = test_y

    np.random.seed(args.seed)
      
    tokenizer = AutoTokenizer.from_pretrained(
        model_name)
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    train_dataset = Dataset(train_encodings, train_y, train_texts)
    test_dataset = Dataset(test_encodings, test_y, test_texts)
    val_dataset = Dataset(val_encodings, val_y, val_texts)

    model = BertMultiSequenceClassification(model_name, 4)
    if model_name == 'roberta-base':
        res = Detoxify('unbiased')
        res = res.model
        pretrained_dict = res.roberta.state_dict()
    elif model_name == 'albert-base-v2':
        res = Detoxify('unbiased-small')
        res = res.model
        pretrained_dict = res.albert.state_dict()
    else:
        raise NotImplementedError

    # Select matching keys
    model_dict = model.bert.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k,
                       v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    assert len(pretrained_dict.keys())>0
    # 3. load the new state dict
    # NB: strict=False because the last layer of model dict (classifier)
    # not in pretrained dict
    model.bert.load_state_dict(pretrained_dict, strict=False)
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.train()

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    curr_acc = 0
    best_acc = -1

    optim = AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.weight).to(device))
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].float().to(device)
            outputs = model(
                input_ids, attention_mask=attention_mask)

            loss = 0
            for num in range(4):
                #NB: Verified that weighting with 0.25 does not make a 
                # difference
                loss += criterion(outputs[:, num], labels[:, num])

            loss.backward()
            optim.step()
        
        model.eval()
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        predictions, true_labels = [], []
        for batch in val_loader:
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['labels'].to(device)
            with torch.no_grad():
                outputs = model(b_input_ids, attention_mask=b_input_mask)

            logits = torch.sigmoid(outputs[:,-1])
            logits=(logits>0.5).float().detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            predictions.extend(logits)
            true_labels.extend(label_ids[:, -1])
        curr_acc = f1_score(true_labels,predictions, average='macro')
        if curr_acc > best_acc:
            best_acc = curr_acc
            best_model_weight = model.state_dict()

    if epochs>0:
        model.load_state_dict(best_model_weight)
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    predictions, true_labels = [], []
    texts = []
    for batch in test_loader:
        b_input_ids = batch['input_ids'].to(device)
        b_input_mask = batch['attention_mask'].to(device)
        b_labels = batch['labels'].to(device)
        with torch.no_grad():
            outputs = model(b_input_ids, attention_mask=b_input_mask)

        logits = outputs.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        predictions.extend(logits)
        true_labels.extend(label_ids)
        texts.extend(batch['texts'])

    print('Evaluation finished')
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    texts = np.array(texts)
    output = pd.DataFrame(
        {"label0": true_labels[:, -1],
         "label1": true_labels[:, 0],
         "label2": true_labels[:, 1],
         "label3": true_labels[:, 2],
         "text": texts,
         "prediction1": predictions[:, 0],
         "prediction2": predictions[:, 1],
         "prediction3": predictions[:, 2],
         "prediction0": predictions[:, 3],
         })
    assert output.shape[0]>0
    output.to_csv('output_main/test_output/contention_ref_{}_{}_contention_{}_bs_{}_seed_{}_cross_{}_weight_{}_size_{}_noise_{}.csv'.format(args.contention_ref,
                                                                                 args.model_name,args.contention,
                                                                                 args.batch_size,
                                                                                 args.seed, args.cross, args.weight,
                                                                                 args.train_size_red,
                                                                                 args.label_noise_prop), index=False)

    # Validation dataset
    test_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    predictions, true_labels = [], []
    texts = []
    for batch in test_loader:
        b_input_ids = batch['input_ids'].to(device)
        b_input_mask = batch['attention_mask'].to(device)
        b_labels = batch['labels'].to(device)
        with torch.no_grad():
            outputs = model(b_input_ids, attention_mask=b_input_mask)

        logits = outputs.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        predictions.extend(logits)
        true_labels.extend(label_ids)
        texts.extend(batch['texts'])

    print('Evaluation finished')
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    texts = np.array(texts)
    output = pd.DataFrame(
        {"label0": true_labels[:, -1],
         "label1": true_labels[:, 0],
         "label2": true_labels[:, 1],
         "label3": true_labels[:, 2],
         "text": texts,
         "prediction1": predictions[:, 0],
         "prediction2": predictions[:, 1],
         "prediction3": predictions[:, 2],
         "prediction0": predictions[:, 3],
         })
    assert output.shape[0]>0
    #output_replication_postdeadline
    output.to_csv('output_main/val_output/contention_ref_{}_{}_contention_{}_bs_{}_seed_{}_cross_{}_weight_{}_size_{}_noise_{}.csv'.format(args.contention_ref,
                                                                                 args.model_name,args.contention,
                                                                                 args.batch_size,
                                                                                 args.seed, args.cross, args.weight,
                                                                                 args.train_size_red,
                                                                                 args.label_noise_prop), index=False)

