from self_attn import *
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


def evaluate(rec_gt=None, img_gt=None, rec_emb=None, img_emb=None, partition='train',
                batch_size=256, device=None, rec_dim=1024, img_dim=768,
                epoch=30,num_layer=2, n_head=4, lr=1e-4, model_path=None, ids=None):
    test_dataset = MyDataset(rec_gt, img_gt, rec_emb, img_emb, partition)

    test_loader = torch.utils.data.DataLoader(test_dataset,
        batch_size=batch_size, shuffle=False, pin_memory=True)

    model = selfAttn(d_rec=rec_dim, d_img=img_dim, nhead=n_head, num_layer=num_layer)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict['model'])
    model.to(device).float()
    model.eval()

    img_emb = np.array([]).reshape((0, 1024))
    rec_emb = np.array([]).reshape((0, 1024))
    for idx, inputs in enumerate(test_loader):
        rec_emb, img_emb, rec_gt, img_gt, tgt = [x.to(device).float() for x in inputs]
        rec_emb.to(device)
        img_emb.to(device)
        # tgt.to(device)
        rec_out, img_out = model(rec_emb, img_emb)
        img_emb = np.vstack([img_emb, img_out.detach().cpu().numpy()])
        rec_emb = np.vstack([rec_emb, rec_out.detach().cpu().numpy()])

    N = img_emb.shape[0]
    recall, median_rank, glob_recall = rank(image_embedding=img_emb[:N], text_embedding=rec_emb[:N],
                                            type_emb='image', ids=ids[:N])
    recall2, median_rank2, glob_recall2 = rank(image_embedding=img_emb[:N], text_embedding=rec_emb[:N],
                                            type_emb='text', ids=ids[:N])

    print("recipe2image:", recall, median_rank, glob_recall)
    print("image2recipe:", recall2, median_rank2, glob_recall2)
    # return recall, median_rank, glob_recall


if __name__ == '__main__':

    partition = 'test'

    recipe_text_path = f'/common/users/sf716/embeddings_{partition}1.pkl'
    rec_gt, img_gt = load_rec_emb(recipe_text_path)
    rec_emb = rec_gt.copy()

    img_fea_path = f'/common/users/sf716/{partition}_img_features.pkl'
    img_emb = load_img_emb(img_fea_path)

    data_path = '/common/users/sf716'

    model_path = '/common/users/sf716/snapshots/vit_5epoch.pt'
    print(model_path)

    os.environ['CUDA_VISIBLE_DEVICES'] = "2"
    device = torch.device("cuda")
    evaluate(rec_gt=rec_gt, img_gt=img_gt, rec_emb=rec_emb, img_emb=img_emb,
                batch_size=512, device=device,
                epoch=100, rec_dim=1024, img_dim=768, num_layer=2, n_head=4, lr=1e-4, ids=list(rec_emb.keys()))
