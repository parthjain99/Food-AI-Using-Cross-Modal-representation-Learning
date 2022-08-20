from transformers import BertTokenizer, BertModel, AdamW, ViTFeatureExtractor, ViTModel
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import pickle
from operator import itemgetter
import torchvision.transforms as transforms
import torch


class ImageLoader(Dataset):
    def __init__(self, path, id_img_dict, ids, tf):
        super(ImageLoader, self).__init__()
        self.imgPath = path
        self.id_img_dict = id_img_dict
        self.ids = ids
        self.tf = tf

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        id = self.ids[index]
        img_list = self.id_img_dict[id]
        imgIdx = 0
        loader_path = [img_list[imgIdx][i] for i in range(4)]
        loader_path = os.path.join(*loader_path)
        path = os.path.join(self.imgPath, loader_path, img_list[imgIdx])
        img = Image.open(path)
        img = torch.tensor(tf(img))
        return id, img


def image_fea(data_loader, device):
    model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
    model.to(device)
    model.eval()

    img_emb = {}
    for idx, sample in enumerate(data_loader):
        id, img = sample
        img = img.to(device)

        out = model(img).pooler_output#.reshape(img.shape[0], -1)
        cur = dict(zip(id, out.detach().cpu().numpy()))
        img_emb = {**img_emb, **cur}

    return img_emb


if __name__ == '__main__':
    partition = 'val'

    with open(os.path.join('/common/users/sf716', f'embeddings_{partition}1.pkl'), 'rb') as f:
        data = pickle.load(f)
    ids = data[2]

    with open('/common/users/sf716/id_img_dict.pkl', 'rb') as f:
        id_img = pickle.load(f)

    imgs = itemgetter(*ids)(id_img)

    id_img_dict = dict(zip(list(id_img.keys()), imgs))
    id_img_dict.pop('98689d3b64', None)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    tf = transforms.Compose([
        transforms.Scale(256),  # rescale the image keeping the original aspect ratio
        transforms.CenterCrop(224),  # we get only the center of that rescaled
        transforms.ToTensor(),
        normalize
    ])

    imgLoader = torch.utils.data.DataLoader(ImageLoader(f'/common/users/sf716/{partition}', id_img_dict, list(id_img_dict.keys()), tf),
                                             shuffle=True, batch_size=32)

    os.environ['CUDA_VISIBLE_DEVICES'] = "3"
    device = torch.device("cuda")

    res = image_fea(imgLoader, device)

    print(res)
    with open(f'/common/users/sf716/{partition}_img_features', 'wb') as f:
        pickle.dump(res, f)



