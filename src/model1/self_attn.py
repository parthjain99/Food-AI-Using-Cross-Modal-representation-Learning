# encoding = utf-8

import torch
from torch.nn.modules.transformer import *
import torch.nn as nn
from torch.utils.data import Dataset
import os
import json
from transformers import BertTokenizer, BertModel, AdamW, ViTFeatureExtractor, ViTModel
from tqdm import *
import pickle
import numpy as np
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms
import sys
import lmdb
import torch.nn.functional as F


class selfAttn(nn.Module):

    def __init__(self, d_rec=1024, d_img=768, nhead=4, num_layer=2):
        super().__init__()
        self.rec_encoder = TransformerEncoder(TransformerEncoderLayer(d_model=d_rec, nhead=nhead, dropout=0.1), num_layers=num_layer)
        self.img_encoder = TransformerEncoder(TransformerEncoderLayer(d_model=d_img, nhead=nhead, dropout=0.1), num_layers=num_layer)
        self.rec_norm = nn.BatchNorm1d(d_rec)
        self.img_norm = nn.BatchNorm1d(d_img)
        self.rec_fc = nn.Linear(in_features=d_rec, out_features=1024)
        self.img_fc = nn.Linear(in_features=d_img, out_features=1024)

    def forward(self, rec_emb, img_emb):
        rec_out = self.rec_encoder(rec_emb)
        rec_out = self.rec_norm(rec_out)
        rec_out = self.rec_fc(rec_out)

        img_out = self.rec_encoder(img_emb)
        img_out = self.rec_norm(img_out)
        img_out = self.rec_fc(img_out)

        return rec_out, img_out


class MyDataset(Dataset):

    def __init__(self, rec_gt, img_gt, rec_emb, img_emb, partition):
        self.rec_gt = rec_gt
        self.img_gt = img_gt
        self.rec_emb = rec_emb
        self.img_emb = img_emb
        self.partition = partition
        self.ids = list(self.img_emb.keys())

    def __getitem__(self, index):
        if self.partition == 'train':
            match = np.random.uniform() > self.mismtch
        elif self.partition == 'val' or self.partition == 'test':
            match = True
        else:
            raise 'Partition name not well defined'

        target = match and 1 or -1

        recipeId = self.ids[index]

        rec_emb = self.rec_emb[recipeId]
        rec_gt = self.rec_gt[recipeId]
        img_emb = self.img_emb[recipeId]
        img_gt = self.img_gt[recipeId]

        if target == -1:
            all_idx = range(len(self.ids))
            rndindex = np.random.choice(all_idx)
            while rndindex == index:
                rndindex = np.random.choice(all_idx)  # pick a random index
            rndId = self.ids[rndindex]
            img_emb = self.img_emb[rndId]
            img_gt = self.img_gt[rndId]

        return rec_emb, img_emb, rec_gt, img_gt, target

    def __len__(self):
        return len(self.ids)


def load_rec_emb(recipe_emb_path):
    with open(recipe_emb_path, 'rb') as f:
        data = pickle.load(f)
    rec_gt, img_gt = dict(zip(data[2], data[0])), dict(zip(data[2], data[1]))
    return rec_gt, img_gt


def load_img_emb(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def train_model(rec_gt=None, img_gt=None, rec_emb=None, img_emb=None, partition='train',
                batch_size=256, device=None, rec_dim=1024, img_dim=768,
                epoch=30,num_layer=2, n_head=4, lr=1e-4):
    train_dataset = MyDataset(rec_gt, img_gt, rec_emb, img_emb, partition)

    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=batch_size, shuffle=True,
        num_workers=10, pin_memory=True)

    attnModel = selfAttn(d_rec=rec_dim, d_img=img_dim, nhead=n_head, num_layer=num_layer)
    attnModel.to(device).float()
    attnModel.train()
    optimizer = AdamW(params=attnModel.parameters(), lr=lr, weight_decay=0.01)

    # triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1 - F.cosine_similarity(x, y), margin=0.3)
    criteria = nn.CosineEmbeddingLoss(margin=0.3)       # this similarity loss contains the negative sample pairs

    for i in tqdm(range(1, epoch + 1)):
        cel_list = []
        for idx, inputs in enumerate(train_loader):
            optimizer.zero_grad()
            rec_emb, img_emb, rec_gt, img_gt, tgt = inputs
            rec_emb.to(device)
            img_emb.to(device)
            # tgt.to(device)
            rec_out, img_out = attnModel(rec_emb, img_emb)
            # loss = triplet_loss(gt, pos, neg)
            loss = criteria(rec_out, img_out, tgt)
            cel_list.append(loss.item())
            loss.backward()
            optimizer.step()
        avg_loss = sum(cel_list) / len(cel_list)
        print(avg_loss)
        if i % 5 == 0:
            state_dict = {
                'model': attnModel.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': i,
                'loss': avg_loss
            }

            torch.save(state_dict, f'/common/users/sf716/snapshots/vit_{i}epoch.pt')


if __name__ == '__main__':

    partition = 'val'
    recipe_text_path = f'/common/users/sf716/embeddings_{partition}1.pkl'
    rec_gt, img_gt = load_rec_emb(recipe_text_path)
    rec_emb = rec_gt.copy()

    img_fea_path = f'/common/users/sf716/{partition}_img_features.pkl'
    img_emb = load_img_emb(img_fea_path)

    data_path = '/common/users/sf716'
    os.environ['CUDA_VISIBLE_DEVICES'] = "3"
    device = torch.device("cuda")
    train_model(rec_gt=rec_gt, img_gt=img_gt, rec_emb=rec_emb, img_emb=img_emb,
                batch_size=512, device=device,
                epoch=300, rec_dim=1024, img_dim=768, num_layer=2, n_head=4, lr=1e-4)
