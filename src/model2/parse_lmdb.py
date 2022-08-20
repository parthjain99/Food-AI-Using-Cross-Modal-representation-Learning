import pickle
import lmdb
import os


def extract():
    keys_path = '/common/users/sf716/train_keys.pkl'
    lmdb_path = '/common/users/sf716/train_lmdb'
    save_path = '/common/users/sf716/train_pair.pkl'
    with open(keys_path, 'rb') as f:
        keys = pickle.load(f)

    env = lmdb.open(lmdb_path, max_readers=1, readonly=True, lock=False,
                    readahead=False, meminit=False)
    key_dict = {}
    for id in keys:
        with env.begin(write=False) as txn:
            serialized_sample = txn.get(id.encode('latin1'))
        sample = pickle.loads(serialized_sample, encoding='latin1')
        img_list = []
        for item in sample['imgs']:
            img_list.append(item['id'])
        key_dict[id] = img_list
        # print(img_list)

    with open(save_path, 'wb') as f:
        pickle.dump(key_dict, f)


if __name__ == '__main__':
    extract()

    # with open('/common/users/sf716/train_pair.pkl', 'rb') as f:
    #     data = pickle.load(f)
    # print(data)