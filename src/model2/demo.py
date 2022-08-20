import json
import pickle
import json


def layer2(path2, path1):

    data2 = json.load(open(path2, 'r'))

    # data1 = json.load(open(path1, 'r'))

    id_img_list = {}
    for item in data2:
        id = item['id']
        img_list = item['images']
        id_img_list[id] = [img['id'] for img in img_list]

    with open('/common/users/sf716/id_img_dict.pkl', 'wb') as f:
        pickle.dump(id_img_list, f)
    print(123)


if __name__ == '__main__':
    l2_path = '/common/users/sf716/layer2.json'
    l1_path = '/common/users/sf716/layer1.json'
    layer2(l2_path, l1_path)