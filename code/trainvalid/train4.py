import argparse
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch
import random
import pickle
from torch.cuda.amp import autocast, GradScaler
import time
from transformers import AutoTokenizer, AutoModel, AutoConfig
import json
import copy
import tokenize
import io
import re
from itertools import combinations

# https://gist.github.com/lorey/eb15a7f3338f959a78cc3661fbc255fe
from bs4 import BeautifulSoup
from markdown import markdown
def markdown_to_text(markdown_string):
    """ Converts a markdown string to plaintext """
    # md -> html -> text since BeautifulSoup can extract text cleanly
    html = markdown(markdown_string)
    # remove code snippets
    html = re.sub(r'<pre>(.*?)</pre>', ' ', html)
    html = re.sub(r'<code>(.*?)</code >', ' ', html)
    # extract text
    soup = BeautifulSoup(html, "html.parser")
    text = ''.join(soup.findAll(text=True))
    return text 

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

class AI4CodeDataset(Dataset):
    def __init__(self, id_list, tokenizer, max_len, label_dict):
        self.id_list=id_list
        self.tokenizer=tokenizer
        self.max_len=max_len
        self.label_dict=label_dict
    def __len__(self):
        return len(self.id_list)
    def __getitem__(self, index):

        # read data
        with open('../../input/train/' + self.id_list[index] + '.json') as json_file:
            data = json.load(json_file)
        cd_id_list = []
        mkd_id_list = []
        for id in data['cell_type']:
            if data['cell_type'][id] == 'code':
                cd_id_list.append(id)
            else:
                mkd_id_list.append(id)

        # randperm mkd cells as augmentation
        perm_mkd_id_list = []
        rand_perm_idx_list = torch.randperm(len(mkd_id_list))
        for i in range(len(mkd_id_list)):
            perm_mkd_id_list.append(mkd_id_list[rand_perm_idx_list[i]])
        mkd_id_list = perm_mkd_id_list

        # special tokens
        mkd_token_id = self.tokenizer.convert_tokens_to_ids('[MKDS]')
        cd_token_id = self.tokenizer.convert_tokens_to_ids('[CDS]')
        bos_token_id = self.tokenizer.bos_token_id
        sep_token_id = self.tokenizer.sep_token_id
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id
        unk_token_id = self.tokenizer.unk_token_id

        # 
        id_list = mkd_id_list + cd_id_list
        cell_token_id_list = [] # list of token_id_list of each cell
        for i in range(len(mkd_id_list)):
            try:
                mkdstr = markdown_to_text(data['source'][mkd_id_list[i]]).strip()
                if len(mkdstr) == 0:
                    oristr = data['source'][mkd_id_list[i]]
                    if oristr[0] == "!" and oristr[1] == "[":
                        for m in range(2, len(oristr)):
                            if oristr[m] == "]":
                                mkdstr = 'embedded ' + oristr[2:m] + ' image'
                                break    
                    elif '<img src' in data['source'][mkd_id_list[i]] or '.png' in data['source'][mkd_id_list[i]] or 'gif' in data['source'][mkd_id_list[i]] or '.jpg' in data['source'][mkd_id_list[i]]:
                        mkdstr = 'embedded image'        
            except:     
                mkdstr = data['source'][mkd_id_list[i]]     
            tokenized = self.tokenizer.encode(text=mkdstr, add_special_tokens=False)
            if len(tokenized) == 0:
                tokenized = [unk_token_id]
            for k in range(len(tokenized)): # avoid special tokens in the original tokens
                if (tokenized[k] == bos_token_id) or (tokenized[k] == sep_token_id) or (tokenized[k] == eos_token_id) or (tokenized[k] == pad_token_id) or (tokenized[k] == mkd_token_id) or (tokenized[k] == cd_token_id):
                    tokenized[k] += 88
            cell_token_id_list.append(tokenized)
        for i in range(len(cd_id_list)):
            try:
                # https://www.kaggle.com/code/haithamaliryan/ai4code-extract-all-functions-variables-names/notebook
                code_text = tokenize.generate_tokens(io.StringIO(data['source'][cd_id_list[i]]).readline)
                cdstr = ' '.join([tok.string for tok in code_text if tok.type in (60, 1, 2, 3)])
                if len(cdstr) == 0:
                    cdstr = "unknown"    
            except:     
                cdstr = data['source'][cd_id_list[i]]     
            tokenized = self.tokenizer.encode(text=cdstr, add_special_tokens=False)
            for k in range(len(tokenized)): # avoid special tokens in the original tokens
                if (tokenized[k] == bos_token_id) or (tokenized[k] == sep_token_id) or (tokenized[k] == eos_token_id) or (tokenized[k] == pad_token_id) or (tokenized[k] == mkd_token_id) or (tokenized[k] == cd_token_id):
                    tokenized[k] += 88
            cell_token_id_list.append(tokenized)

        # 
        cell_max_len = int((self.max_len - (len(id_list) + 5)) / len(id_list)) # starting average trunc len per cell
        cell_len_list = [] # final trunc len per cell
        n_free = 0
        for i in range(len(id_list)):
            if cell_max_len <= len(cell_token_id_list[i]):
                cell_len_list.append(cell_max_len)
            else:
                cell_len_list.append(len(cell_token_id_list[i]))
                n_free += (cell_max_len - len(cell_token_id_list[i])) 
        n_free += (self.max_len - cell_max_len * len(id_list) - (len(id_list) + 5))

        if n_free > 0:
            n_free_reduce = 1
            while 1:
                if (n_free == 0) or (n_free_reduce == 0):
                    break
                n_free_reduce = 0
                for i in range(len(id_list)):   
                    if cell_len_list[i] < len(cell_token_id_list[i]):
                        cell_len_list[i] += 1
                        n_free -= 1 
                        n_free_reduce += 1
                        if n_free == 0:
                            break           

        #
        mkd_start_end_list = [] # list of start_end indices of each mkd cell
        cd_start_end_list = [] # list of start_end indices of each cd cell
        token_id_list = [] # final token_id_list as input_ids
        token_id_list.append(bos_token_id)
        token_id_list.append(mkd_token_id)
        for i in range(len(mkd_id_list)):
            start = len(token_id_list)
            token_id_list += cell_token_id_list[i][:cell_len_list[i]]
            end = len(token_id_list)
            token_id_list.append(mkd_token_id)
            mkd_start_end_list.append((start, end))
        token_id_list.append(sep_token_id)
        token_id_list.append(cd_token_id)
        for i in range(len(mkd_id_list), len(id_list)):
            start = len(token_id_list)
            token_id_list += cell_token_id_list[i][:cell_len_list[i]]
            end = len(token_id_list)
            token_id_list.append(cd_token_id)
            cd_start_end_list.append((start, end))
        token_id_list.append(eos_token_id)       

        # get targets
        target_list = []
        for i in range(len(mkd_id_list)):        
            target_list.append(self.label_dict[self.id_list[index]][mkd_id_list[i]])
        for i in range(len(cd_id_list)):        
            target_list.append(self.label_dict[self.id_list[index]][cd_id_list[i]])

        # get rank pairs and targets
        rank_pair_list = list(combinations(range(len(id_list)), 2))
        rank_target_list = []
        for i in range(len(rank_pair_list)):
            if self.label_dict[self.id_list[index]][id_list[rank_pair_list[i][0]]] < self.label_dict[self.id_list[index]][id_list[rank_pair_list[i][1]]]:
                rank_target_list.append(-1)
            else:
                rank_target_list.append(1)
         
        start_end_list = mkd_start_end_list + cd_start_end_list

        return token_id_list, start_end_list, target_list, pad_token_id, rank_pair_list, rank_target_list

def collate_fn(batch):
    pad_token_id = int(batch[0][3])
    len_list = []
    for i in range(len(batch)):
        len_list.append(len(batch[i][0]))
    max_len = max(len_list)
    #print(max_len)
    input_ids_list = []
    attention_mask_list = []
    start_end_list = []
    target_list = []
    rank_pair_list = []
    rank_target_list = []
    for i in range(len(batch)):
        input_ids = batch[i][0] + [pad_token_id]*(max_len-len(batch[i][0]))
        attention_mask = [1]*len(batch[i][0]) + [0]*(max_len-len(batch[i][0]))
        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        start_end_list.append(batch[i][1])
        target_list += batch[i][2]
        rank_pair_list.append(batch[i][4])
        rank_target_list += batch[i][5]
    input_ids_list = np.stack(input_ids_list)
    input_ids_list = torch.tensor(input_ids_list, dtype=torch.long)
    attention_mask_list = np.stack(attention_mask_list)
    attention_mask_list = torch.tensor(attention_mask_list, dtype=torch.long)
    target_list = torch.tensor(target_list, dtype=torch.float32)
    rank_target_list = torch.tensor(rank_target_list, dtype=torch.long)
    return input_ids_list, attention_mask_list, start_end_list, target_list, rank_pair_list, rank_target_list

class AI4CodeDebertav3largeModel(nn.Module):
    def __init__(self, model_path, config, tokenizer, pretrained=False):
        super().__init__()
        self.config = config
        if pretrained:
            self.model = AutoModel.from_pretrained(model_path, config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)
        self.model.resize_token_embeddings(len(tokenizer))
        self.classifier = nn.Linear(self.config.hidden_size, 1)      
    def forward(self, input_ids, attention_mask, start_end_list):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits_list = []
        for n in range(outputs[0].shape[0]):
            temp_logits_list = []
            for i in range(len(start_end_list[n])):
                mean_pooled = torch.mean(outputs[0][n, start_end_list[n][i][0]:start_end_list[n][i][1], :], dim=0)
                logits = self.classifier(mean_pooled)
                temp_logits_list.append(logits)
            logits_list.append(temp_logits_list)
        return logits_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    args = parser.parse_args()
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend="nccl")
    args.device = device

    seed = 1041
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # prepare input
    fold = 0
    import pickle
    with open('../splits/split1/train_id_cv_list.pickle', 'rb') as f:
        id_list = pickle.load(f)[fold]
    print(len(id_list))

    df_orders = pd.read_csv('../../input/train_orders.csv', dtype={'id': str, 'cell_order': str})
    id_list_orders = df_orders['id'].values
    cell_order_list = df_orders['cell_order'].values
    label_dict = {}
    for i in range(len(id_list_orders)):
        label_dict[id_list_orders[i]] = {}
        order_list = cell_order_list[i].split(' ')
        for j in range(len(order_list)):
            label_dict[id_list_orders[i]][order_list[j]] = j/(len(order_list)-1)

    # hyperparameters
    learning_rate = 0.000001
    max_len = 2048
    batch_size = 2
    num_epoch = 5
    num_polyak = 48
    model_path = 'microsoft/deberta-v3-large'

    # build model
    if args.local_rank != 0:
        torch.distributed.barrier()
    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    num_added_toks = tokenizer.add_tokens(['[MKDS]', '[CDS]'])
    print('We have added', num_added_toks, 'tokens')
    model = AI4CodeDebertav3largeModel(model_path, config, tokenizer, pretrained=True)
    if args.local_rank == 0:
        torch.distributed.barrier()
    if args.local_rank == 0:
        model.load_state_dict(torch.load('weights/weights3'))
    model.to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    # training
    train_datagen = AI4CodeDataset(id_list, tokenizer, max_len, label_dict)
    train_sampler = DistributedSampler(train_datagen)
    train_generator = DataLoader(dataset=train_datagen,
                                 collate_fn=collate_fn,
                                 sampler=train_sampler,
                                 batch_size=batch_size,
                                 num_workers=8,
                                 pin_memory=False)

    if args.local_rank == 0:
        start_time = time.time()

    scaler = GradScaler()
    for ep in range(4,num_epoch):
        losses = AverageMeter()
        model.train()
        for j, (batch_input_ids, batch_attention_mask, batch_start_end_list, batch_targets, batch_rank_pair_list, batch_rank_targets) in enumerate(train_generator):
            batch_input_ids = batch_input_ids.to(args.device)
            batch_attention_mask = batch_attention_mask.to(args.device)
            batch_targets = batch_targets.to(args.device)
            batch_rank_targets = batch_rank_targets.to(args.device)

            with autocast():
                loss_bce = 0
                logits_list = model(batch_input_ids, batch_attention_mask, batch_start_end_list)
                curr_idx = 0
                for n in range(len(logits_list)):
                    for m in range(len(logits_list[n])):
                        loss_bce += nn.BCEWithLogitsLoss()(logits_list[n][m], batch_targets[curr_idx].unsqueeze(0))
                        curr_idx += 1
                loss_bce /= len(batch_targets)

                loss_rank = 0
                curr_idx = 0
                for n in range(len(logits_list)):
                    for m in range(len(batch_rank_pair_list[n])):
                        loss_rank += nn.MarginRankingLoss(margin=0.5)(logits_list[n][batch_rank_pair_list[n][m][0]], logits_list[n][batch_rank_pair_list[n][m][1]], batch_rank_targets[curr_idx].unsqueeze(0))
                        curr_idx += 1
                loss_rank /= len(batch_rank_targets)

                loss = 0*loss_bce + loss_rank

            losses.update(loss.item(), batch_input_ids.size(0))

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            if args.local_rank == 0:
                if j==len(train_generator)-num_polyak:
                    averaged_model = copy.deepcopy(model.module.state_dict())
                if j>len(train_generator)-num_polyak:
                    for k in averaged_model.keys():
                        averaged_model[k].data += model.module.state_dict()[k].data
                if j==len(train_generator)-1:
                    for k in averaged_model.keys():
                        averaged_model[k].data = averaged_model[k].data / float(num_polyak)

            if args.local_rank == 0:
                print('\r',end='',flush=True)
                message = '%s %5.1f %6.1f    |     %0.3f     |' % ("train",j/len(train_generator)+ep,ep,losses.avg)
                print(message , end='',flush=True)

        if args.local_rank == 0:
            print('epoch: {}, train_loss: {}'.format(ep, losses.avg), flush=True)

    if args.local_rank == 0:
        out_dir = 'weights/'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        torch.save(model.module.state_dict(), out_dir+'weights{}'.format(ep))
        torch.save(averaged_model, out_dir+'weights{}_polyak'.format(ep))

    if args.local_rank == 0:
        end_time = time.time()
        print(end_time-start_time)

if __name__ == "__main__":
    main()
