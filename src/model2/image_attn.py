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


class RecipeMultiHeadAttn(nn.Module):

    def __init__(self, d_model=2048, nhead=4, dropout=0.1, kdim=768, vdim=768, num_layer=2):
        super().__init__()
        self.decoderLayer = TransformerDecoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        # self.decoderLayer.multihead_attn = MultiheadAttention(embed_dim=d_model, num_heads=nhead, kdim=kdim, vdim=vdim, batch_first=True)
        self.decoder = TransformerDecoder(decoder_layer=self.decoderLayer, num_layers=num_layer)
        # self.resnet = models.resnet50(pretrained=True)
        # self.resnet.requires_grad_(False)
        # self.extractors = nn.Sequential(*list(self.resnet.children())[:-1])
        self.extractors = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        for param in self.extractors.parameters():
            param.requires_grad = False
        self.fc = nn.Linear(in_features=d_model, out_features=768, bias=True)

    def forward(self, img, memory):
        # extractor = nn.Sequential(*list(self.resnet.children())[:-1])
        seq = self.extractors(img).pooler_output.reshape(img.shape[0], 1, -1)
        # print(seq.shape)
        # print(seq.shape, memory.shape)
        out = self.decoder(seq, memory)
        out = self.fc(out)
        out = out.squeeze(1)
        return out


def default_loader(path):
    try:
        im = Image.open(path).convert('RGB')
        return im
    except:
        print(..., file=sys.stderr)
        return Image.new('RGB', (224, 224), 'white')


class ImagerLoader(Dataset):

    def __init__(self, img_path, transform=None, target_transform=None, remove_ids=None,
                 loader=default_loader, square=False, recipe_emb=None, partition=None, data_path=None, all_text_emb_dict=None):

        self.partition = partition
        self.square = square
        self.imgPath = img_path
        self.mismtch = 0.8
        self.maxInst = 20
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.recipe_emb = recipe_emb
        self.ids = list(recipe_emb.keys())
        self.all_text_emb_dict = all_text_emb_dict

        self.env = lmdb.open(os.path.join(data_path, partition + '_lmdb'), max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)

    def __getitem__(self, index):
        recipeId = self.ids[index]
        # we force 80 percent of them to be a mismatch
        if self.partition == 'train' or self.partition == 'val':
            match = np.random.uniform() > self.mismtch
        elif self.partition == 'val' or self.partition == 'test':
            match = True
        else:
            raise 'Partition name not well defined'

        target = match and 1 or -1

        emb = self.recipe_emb[recipeId]['text_emb']

        with self.env.begin(write=False) as txn:
            serialized_sample = txn.get(self.ids[index].encode('latin1'))
        sample = pickle.loads(serialized_sample, encoding='latin1')
        imgs = sample['imgs']

        # image
        if target == 1:
            if self.partition == 'train' or self.partition == 'val':
                # We do only use the first five images per recipe during training
                imgIdx = np.random.choice(range(min(5, len(imgs))))
            else:
                imgIdx = 0

            loader_path = [imgs[imgIdx]['id'][i] for i in range(4)]
            loader_path = os.path.join(*loader_path)
            path = os.path.join(self.imgPath, loader_path, imgs[imgIdx]['id'])
        else:
            # we randomly pick one non-matching image
            all_idx = range(len(self.ids))
            rndindex = np.random.choice(all_idx)
            while rndindex == index:
                rndindex = np.random.choice(all_idx)  # pick a random index

            with self.env.begin(write=False) as txn:
                serialized_sample = txn.get(self.ids[rndindex].encode('latin1'))

            rndsample = pickle.loads(serialized_sample, encoding='latin1')
            rndimgs = rndsample['imgs']

            if self.partition == 'train':  # if training we pick a random image
                # We do only use the first five images per recipe during training
                imgIdx = np.random.choice(range(min(5, len(rndimgs))))
            else:
                imgIdx = 0

            loader_path = [rndimgs[imgIdx]['id'][i] for i in range(4)]
            loader_path = os.path.join(*loader_path)
            path = os.path.join(self.imgPath, loader_path, rndimgs[imgIdx]['id'])

        # load image
        img = self.loader(path)
        img = self.transform(img)

        # ground truth
        gt = self.all_text_emb_dict[recipeId]

        return img, emb, target, gt

    def __len__(self):
        return len(self.ids)


def load_rec_text(recipe_emb_path):
    with open(recipe_emb_path, 'rb') as f:
        data = pickle.load(f)
    return data
    # print(data)


def load_text_emp(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    ret = dict(zip(data[2], data[0]))
    return ret


def train_model(img_path, loader=default_loader, square=False, recipe_emb=None,
                partition=None, data_path=None, batch_size=1, device=None, all_text_emb_dict=None,
                epoch=100, d_model=768, num_layer=2, n_head=4, lr=1e-4, pretrained_model=None):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        ImagerLoader(img_path=img_path, transform=transforms.Compose([
            transforms.Scale(256),  # rescale the image keeping the original aspect ratio
            transforms.CenterCrop(224),  # we get only the center of that rescaled
            transforms.ToTensor(),
            normalize
        ]), data_path=data_path, recipe_emb=recipe_emb, square=square, partition=partition, all_text_emb_dict=all_text_emb_dict),
        batch_size=batch_size, shuffle=True,
        num_workers=10, pin_memory=True)

    attnModel = RecipeMultiHeadAttn(d_model=d_model, nhead=n_head, dropout=0.1, kdim=768, vdim=768, num_layer=num_layer)
    # for name, params in attnModel.named_parameters():
    #     print(name, params.requires_grad)
    attnModel.to(device).float()
    attnModel.train()
    optimizer = AdamW(params=attnModel.parameters(), lr=lr, weight_decay=0.01)
    if pretrained_model is not None:
        state_dict = torch.load(pretrained_model)
        attnModel.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
    criteria = nn.CosineEmbeddingLoss()

    for i in tqdm(range(1, epoch + 1)):
        cel_list = []
        for idx, inputs in enumerate(train_loader):
            optimizer.zero_grad()
            img, memory, target, gt = [x.to(device).float() for x in inputs]
            out = attnModel(img, memory)
            loss = criteria(out, gt, target)
            cel_list.append(loss.item())
            loss.backward()
            optimizer.step()
        avg_loss =sum(cel_list) / len(cel_list)
        print(avg_loss)
        if i % 10 == 0:
            state_dict = {
                'model': attnModel.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': i,
                'loss': avg_loss
            }

            torch.save(state_dict, f'/common/users/sf716/snapshots/e4_08_512/vit_{i}epoch.pt')


if __name__ == '__main__':
    # with open(os.path.join('/common/users/sf716/train_keys.pkl'), 'rb') as f:
    #     curids = pickle.load(f)

    partition = 'val'
    recipe_text_path = f'/common/users/sf716/recipe_{partition}_emb.pkl'
    rec_emb = load_rec_text(recipe_text_path)
    max_len = max([v['text_emb'].shape[0] for k, v in rec_emb.items()])
    for id in rec_emb:
        cur = rec_emb[id]['text_emb']
        sub = np.zeros((max_len - cur.shape[0], 768))
        rec_emb[id]['text_emb'] = np.vstack((cur, sub))
    print(max_len)
    # for id in rec_list:
    #     rec_emb.pop(id, None)

    # all_text_emb_path = f'/common/home/sf716/Downloads/dataset/embeddings_{partition}1.pkl'
    # all_text_emb_dict = load_text_emp(all_text_emb_path)

    text_gt_emb_path = f'/common/users/sf716/{partition}_gt_emb.pkl'
    with open(text_gt_emb_path, 'rb') as f:
        gt_emb = pickle.load(f)

    pretrained_model = None  #'../snapshots/vit_100epoch.pt'
    img_path = f'/common/users/sf716/{partition}'
    data_path = '/common/users/sf716'
    os.environ['CUDA_VISIBLE_DEVICES'] = "3"
    device = torch.device("cuda")
    train_model(img_path=img_path, recipe_emb=rec_emb, partition=partition,
                data_path=data_path, batch_size=512, device=device, all_text_emb_dict=gt_emb,
                epoch=300, d_model=768, num_layer=2, n_head=4, lr=1e-4, pretrained_model=pretrained_model)
