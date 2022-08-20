from triplet import *
import random


def rank(image_embedding, text_embedding, type_emb, ids):
    # text_ids=image_test[2]
    text_ids = np.array(ids)
    round_size = 1000
    rounds = 10
    seed = 11
    st = random.getstate()
    random.seed(seed)

    text_id_list = text_ids[np.argsort(text_ids)]
    median_rank_per_round, rank_list_per_round = [], []
    glob_recall = {1: 0.0, 5: 0.0, 10: 0.0}

    for _ in range(rounds):
        ids = random.sample(range(0, len(text_id_list)), round_size)
        sub_image_emb = image_embedding[ids, :]
        sub_text_emb = text_embedding[ids, :]
        if type_emb == 'image':
            sims = np.dot(sub_image_emb, sub_text_emb.T)  # image2recipe
        else:
            sims = np.dot(sub_text_emb, sub_image_emb.T)  # recipe2image

        median_rank_list = []
        recall = {1: 0.0, 5: 0.0, 10: 0.0}
        for i in range(round_size):
            sim = sims[i, :]
            sorting = np.argsort(sim)[::-1].tolist()
            pos = sorting.index(i)
            recall[1] += 1 if (pos + 1) == 1 else 0
            recall[5] += 1 if (pos + 1) <= 5 else 0
            recall[10] += 1 if (pos + 1) <= 10 else 0
            median_rank_list.append(pos + 1)

        recall = {k: (v / round_size) for k, v in recall.items()}
        rank_list_per_round.append(median_rank_list)
        median_rank = np.median(median_rank_list)

        glob_recall = {k: v + recall[k] for k, v in glob_recall.items()}
        median_rank_per_round.append(median_rank)

    glob_recall = {k: (v / rounds) for k, v in glob_recall.items()}
    random.setstate(st)

    return recall, median_rank, glob_recall


def eva(img_path, square=False, recipe_emb=None, partition=None, data_path=None, batch_size=1,
        device=None, all_text_emb_dict=None, d_model=768, model_path=None, num_layer=2, n_head=4):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    test_loader = torch.utils.data.DataLoader(
        ImagerLoader(img_path=img_path, transform=transforms.Compose([
            transforms.Scale(256),  # rescale the image keeping the original aspect ratio
            transforms.CenterCrop(224),  # we get only the center of that rescaled
            transforms.ToTensor(),
            normalize
        ]), data_path=data_path, recipe_emb=recipe_emb, square=square, partition=partition, all_text_emb_dict=all_text_emb_dict),
        batch_size=batch_size, shuffle=False,
        num_workers=10, pin_memory=True)

    model = RecipeMultiHeadAttn(d_model=d_model, nhead=n_head, dropout=0.1, kdim=768, vdim=768, num_layer=num_layer)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict['model'])
    model.to(device).float()
    model.eval()

    img_emb = np.array([]).reshape((0, 768))
    for idx, inputs in enumerate(test_loader):
        # img, memory, target, gt = [x.to(device).float() for x in inputs]
        pos_img, neg_img, memory, gt = [x.to(device).float() for x in inputs]
        out = model(pos_img, memory)
        img_emb = np.vstack([img_emb, out.detach().cpu().numpy()])

    N = img_emb.shape[0]
    recall, median_rank, glob_recall = rank(image_embedding=img_emb[:N], text_embedding=np.array(list(all_text_emb_dict.values())[:N]),
                                            type_emb='image', ids=list(all_text_emb_dict.keys())[:N])
    print(recall, median_rank, glob_recall)
    return recall, median_rank, glob_recall


if __name__ == '__main__':

    partition = 'test'
    recipe_text_path = f'/common/users/sf716/vp_{partition}_emb.pkl'
    rec_emb = load_rec_text(recipe_text_path)
    rec_emb.pop('5eba8f2ef6', None)
    max_len = max([np.array(v['text_emb']).shape[0] for k, v in rec_emb.items()])
    print(max_len)
    for id in rec_emb:
        cur = np.array(rec_emb[id]['text_emb']).reshape(-1, 768)
        sub = np.zeros((max_len - cur.shape[0], 768))
        rec_emb[id]['text_emb'] = np.vstack((cur, sub))
    # print(max_len)

    text_gt_emb_path = f'/common/users/sf716/vp_{partition}_gt_emb.pkl'
    with open(text_gt_emb_path, 'rb') as f:
        gt_emb = pickle.load(f)

    model_path = '/common/users/sf716/snapshots/tri_e4_08_512/vit_10epoch.pt'
    # model_path = '/common/users/sf716/snapshots/tri_e4_08_512/vit_10epoch.pt'
    print(model_path)

    img_path = f'/common/users/sf716/{partition}'
    data_path = '/common/users/sf716'
    os.environ['CUDA_VISIBLE_DEVICES'] = "3"
    device = torch.device("cuda")
    eva(img_path=img_path, recipe_emb=rec_emb, partition=partition, data_path=data_path, batch_size=256,
        device=device, all_text_emb_dict=gt_emb, d_model=768, model_path=model_path, num_layer=2, n_head=4)
