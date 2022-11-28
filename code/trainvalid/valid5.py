import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
import pickle
import time
from torch.cuda.amp import autocast, GradScaler
import random
import json
from bisect import bisect
import tokenize
import io
import re

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

def count_inversions(a):
    inversions = 0
    sorted_so_far = []
    for i, u in enumerate(a):
        j = bisect(sorted_so_far, u)
        inversions += i - j
        sorted_so_far.insert(j, u)
    return inversions

def kendall_tau(ground_truth, predictions):
    total_inversions = 0
    total_2max = 0  # twice the maximum possible inversions across all instances
    for gt, pred in zip(ground_truth, predictions):
        ranks = [gt.index(x) for x in pred]  # rank predicted order in terms of ground truth
        total_inversions += count_inversions(ranks)
        n = len(gt)
        total_2max += n * (n - 1)
    return 1 - 4 * total_inversions / total_2max

class AI4CodeDataset(Dataset):
    def __init__(self, id_list, tokenizer, max_len):
        self.id_list=id_list
        self.tokenizer=tokenizer
        self.max_len=max_len
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

        start_end_list = mkd_start_end_list + cd_start_end_list  
          
        return token_id_list, start_end_list, id_list, pad_token_id

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
    id_list = []
    for i in range(len(batch)):
        input_ids = batch[i][0] + [pad_token_id]*(max_len-len(batch[i][0]))
        attention_mask = [1]*len(batch[i][0]) + [0]*(max_len-len(batch[i][0]))
        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        start_end_list.append(batch[i][1])
        id_list.append(batch[i][2])
    input_ids_list = np.stack(input_ids_list)
    input_ids_list = torch.tensor(input_ids_list, dtype=torch.long)
    attention_mask_list = np.stack(attention_mask_list)
    attention_mask_list = torch.tensor(attention_mask_list, dtype=torch.long)
    return input_ids_list, attention_mask_list, start_end_list, id_list

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
    #@autocast()     
    def forward(self, input_ids, attention_mask, start_end_list):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits_list = []
        for n in range(outputs[0].shape[0]):
            for i in range(len(start_end_list[n])):
                mean_pooled = torch.mean(outputs[0][n, start_end_list[n][i][0]:start_end_list[n][i][1], :], dim=0)
                logits = self.classifier(mean_pooled)
                logits_list.append(logits)
        return logits_list

def main():

    start_time = time.time()

    fold = 0
    import pickle
    with open('../splits/split1/valid_id_cv_list.pickle', 'rb') as f:
        id_list = pickle.load(f)[fold]
    print(len(id_list))

    cell_len_list = []
    for i in tqdm(range(len(id_list))):
        cell_len_list1 = 0
        with open('../../input/train/' + id_list[i] + '.json') as json_file:
            data = json.load(json_file)
            for id in data['cell_type']:
                cell_len_list1 += len(data['source'][id])
        cell_len_list.append(cell_len_list1)
    cell_len_list = np.array(cell_len_list)
    id_list = np.array(id_list)
    sorted_idx = np.argsort(cell_len_list)[::-1]
    id_list = id_list[sorted_idx] 
    id_list = list(id_list)

    df_orders = pd.read_csv('../../input/train_orders.csv', index_col='id', squeeze=True).str.split()

    max_len = 4096
    batch_size = 4
    model_path = 'microsoft/deberta-v3-large'

    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    num_added_toks = tokenizer.add_tokens(['[MKDS]', '[CDS]'])
    print('We have added', num_added_toks, 'tokens')
    model = AI4CodeDebertav3largeModel(model_path, config, tokenizer, pretrained=False)
    model.load_state_dict(torch.load('weights/weights5_polyak'))
    model.cuda()
    #if torch.cuda.device_count() > 1:
    #    model = torch.nn.DataParallel(model)
    model.eval()

    test_datagen = AI4CodeDataset(id_list, tokenizer, max_len)
    test_generator = DataLoader(dataset=test_datagen,
                                collate_fn=collate_fn,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=8,
                                pin_memory=False)

    pred_order_list = []
    for j, (batch_input_ids, batch_attention_mask, batch_start_end_list, batch_id_list) in tqdm(enumerate(test_generator), total=len(test_generator)):

        with torch.no_grad():
            batch_input_ids = batch_input_ids.cuda()
            batch_attention_mask = batch_attention_mask.cuda()
            with autocast():
                logits_list = model(batch_input_ids, batch_attention_mask, batch_start_end_list)            
            
            curr_idx = 0
            for n in range(batch_input_ids.size(0)):
                pred_score_list = []
                pred_id_list = batch_id_list[n]
                for m in range(len(pred_id_list)):
                    pred_score = logits_list[curr_idx].cpu().data.numpy()[0]
                    pred_score_list.append(pred_score)
                    curr_idx += 1
                pred_id_list = np.array(pred_id_list)
                pred_score_list = np.array(pred_score_list)
                sorted_idx = np.argsort(pred_score_list)
                pred_order_list.append(list(pred_id_list[sorted_idx]))                

    df_pred = pd.DataFrame(data={'cell_order': pred_order_list}, index=id_list)

    print()
    print('score: ', kendall_tau(list(df_orders.loc[df_pred.index].values), pred_order_list))

    print()
    end_time = time.time()
    print(end_time-start_time)

if __name__ == "__main__":
    main()
