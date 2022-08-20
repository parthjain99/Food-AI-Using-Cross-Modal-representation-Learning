# encoding = utf-8

import torch
from torch.nn.modules.transformer import *
import torch.nn as nn
import os
import json
from transformers import BertTokenizer, BertModel, ViTFeatureExtractor
from tqdm import *
import pickle
import numpy as np
import lmdb


def parseLayer1(layer1_path, device, partition='train'):
    # env = lmdb.open('/common/users/sf716/' + partition + '_lmdb', max_readers=1, readonly=True, lock=False,
    #                 readahead=False, meminit=False)
    # txn = env.begin(write=False)

    with open(f'/common/home/sf716/Downloads/dataset/embeddings_{partition}1.pkl', 'rb') as f:
        emb_sample = pickle.load(f)
    id_set = set(emb_sample[2])

    model = BertModel.from_pretrained("bert-base-uncased")
    model.to(device)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    ids, texts = [], []
    layer1 = json.load(open(layer1_path, 'r'))
    for i, entry in tqdm(enumerate(layer1)):
        id = entry['id']
        if id not in id_set:
            continue
        # serialized_sample = txn.get(id.encode('latin1'))
        # if serialized_sample is None:
        #     continue
        ids.append(id)
        title = entry['title']
        ingredients = entry['ingredients']
        instructions = entry['instructions']
        max_len = 10
        text = [title] + [ingr['text'] for ingr in ingredients[:min(max_len, len(ingredients))]] + [inst['text'] for inst in instructions[:min(max_len, len(instructions))]]
        # emb = text2emb(text, model, tokenizer, id, device)
        texts.append(" ".join(text))
        # if (i + 1) / 1000 > 1:
        #     break
    i = 0
    bs = 16
    embs = None
    while i + bs < len(ids):
        print(i, len(ids))
        emb = text2emb(texts[i:i + bs], model, tokenizer, device)
        if embs is None:
            embs = np.array(emb)
        else:
            embs = np.concatenate([embs, emb], axis=0)
        i += bs
    emb = text2emb(texts[i:], model, tokenizer, device)
    embs = np.concatenate([embs, emb], axis=0)

    recipe_dict = dict(zip(ids, embs))
    with open(os.path.join('/common/users/sf716', 'vp_'+partition+'_gt_emb.pkl'), 'wb') as f:
        pickle.dump(recipe_dict, f)


def text2emb(text, model, tokenizer, device):
    encoding_dict = tokenizer.batch_encode_plus(
        batch_text_or_text_pairs=text,  # Sentence to encode.
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        # max_length=20,  # Pad & truncate all sentences.
        pad_to_max_length=True,
        return_attention_mask=True,  # Construct attn. masks
        return_tensors='pt',  # Return pytorch tensors.
    )
    input_ids = encoding_dict["input_ids"].to(device)
    attn_mask = encoding_dict['attention_mask'].to(device)

    try:
        cls = model(input_ids, token_type_ids=None, attention_mask=attn_mask)[1]
    except Exception as e:
        print(e)
        print(input_ids.shape)
        return np.zeros([len(text), 768])
    return cls.detach().cpu().numpy()


if __name__ == '__main__':
    layer1_path = "/common/users/sf716/layer1.json"
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    device = torch.device("cuda")
    parseLayer1(layer1_path, device, partition='test')

    # with open('/common/users/sf716/recipe_train_emb.pkl', 'rb') as f:
    #     data = pickle.load(f)
    #
    # print(data)
