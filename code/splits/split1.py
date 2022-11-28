import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import pickle
from sklearn.model_selection import KFold

df = pd.read_csv('../../input/train_ancestors.csv', dtype={'id': str, 'ancestor_id': str})
id_list = df['id'].values
ancestor_list = df['ancestor_id'].values

unique_ancestor_list = sorted(list(set(ancestor_list)))
ancestor2id = {}
for i in range(len(id_list)):
    ancestor = ancestor_list[i]
    if ancestor not in ancestor2id:
        ancestor2id[ancestor] = [id_list[i]]
    else:
        ancestor2id[ancestor].append(id_list[i])
print(len(unique_ancestor_list), len(id_list))

# create 10 folds
unique_ancestor_list = np.array(unique_ancestor_list)
k = 10
split_seed = 1
train_id_cv_list = []
valid_id_cv_list = []
kf = KFold(n_splits=k, shuffle=True, random_state=split_seed)

for train_index, valid_index in kf.split(unique_ancestor_list):
    train_id_cv_list1 = []
    valid_id_cv_list1 = []

    train_ancestor_list = unique_ancestor_list[train_index]
    valid_ancestor_list = unique_ancestor_list[valid_index]

    for ancestor in train_ancestor_list:
        train_id_cv_list1 += ancestor2id[ancestor]                           
    train_id_cv_list.append(train_id_cv_list1)

    for ancestor in valid_ancestor_list:
        valid_id_cv_list1 += ancestor2id[ancestor]                           
    valid_id_cv_list.append(valid_id_cv_list1)

    print(len(train_index), len(valid_index))

print(len(train_id_cv_list[0]), len(valid_id_cv_list[0]))
print(len(train_id_cv_list[1]), len(valid_id_cv_list[1]))
print(len(train_id_cv_list[2]), len(valid_id_cv_list[2]))
print(len(train_id_cv_list[3]), len(valid_id_cv_list[3]))
print(len(train_id_cv_list[4]), len(valid_id_cv_list[4]))
print(len(train_id_cv_list[5]), len(valid_id_cv_list[5]))
print(len(train_id_cv_list[6]), len(valid_id_cv_list[6]))
print(len(train_id_cv_list[7]), len(valid_id_cv_list[7]))
print(len(train_id_cv_list[8]), len(valid_id_cv_list[8]))
print(len(train_id_cv_list[9]), len(valid_id_cv_list[9]))

out_dir = 'split1/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
with open(out_dir+'train_id_cv_list.pickle', 'wb') as f:
    pickle.dump(train_id_cv_list, f, protocol=4)
with open(out_dir+'valid_id_cv_list.pickle', 'wb') as f:
    pickle.dump(valid_id_cv_list, f, protocol=4)


