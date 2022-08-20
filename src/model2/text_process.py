# encoding = utf-8

import torch
from torch.nn.modules.transformer import *
import torch.nn as nn
import os
import json
from transformers import BertTokenizer, BertModel
from tqdm import *
import pickle
import numpy as np
import lmdb


def parseLayer1(layer1_path, device, partition='train'):
    # env = lmdb.open('/common/users/sf716/'+partition+'_lmdb', max_readers=1, readonly=True, lock=False,
    #                 readahead=False, meminit=False)
    # txn = env.begin(write=False)

    with open(f'/common/home/sf716/Downloads/dataset/embeddings_{partition}1.pkl', 'rb') as f:
        emb_sample = pickle.load(f)
    id_set = set(emb_sample[2])

    model = BertModel.from_pretrained("bert-base-uncased")
    model.to(device)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    recipe_dict = {}
    layer1 = json.load(open(layer1_path, 'r'))
    max_len = 0
    for i, entry in tqdm(enumerate(layer1)):
        id = entry['id']
        if id not in id_set:
            continue
        # serialized_sample = txn.get(id.encode('latin1'))
        # if serialized_sample is None:
        #     continue
        recipe_dict[id] = {}
        recipe_dict[id]["partition"] = entry['partition']
        title = entry['title']
        ingredients = entry['ingredients']
        instructions = entry['instructions']
        text = [title] + [ingr['text'] for ingr in ingredients] + [inst['text'] for inst in instructions]
        emb = text2emb(text, model, tokenizer, id, device)
        recipe_dict[id]['text_emb'] = emb
        # if (i + 1) % 500 == 0:
        #     print(f"{i} th sample")
        # if (i+1)/1000 > 1:
        #     break

    with open(os.path.join('/common/users/sf716', 'vp_'+partition+'_emb.pkl'), 'wb') as f:
        pickle.dump(recipe_dict, f)

    # text = ['Worlds Best Mac and Cheese', '6 ounces penne', '2 cups Beechers Flagship Cheese Sauce (recipe follows)', '1 ounce Cheddar, grated (1/4 cup)',
    #         '1 ounce Gruyere cheese, grated (1/4 cup)', '1/4 to 1/2 teaspoon chipotle chili powder (see Note)', '1/4 cup (1/2 stick) unsalted butter', '1/3 cup all-purpose flour',
    #         '3 cups milk', '14 ounces semihard cheese (page 23), grated (about 3 1/2 cups)', '2 ounces semisoft cheese (page 23), grated (1/2 cup)', '1/2 teaspoon kosher salt',
    #         '1/4 to 1/2 teaspoon chipotle chili powder', '1/8 teaspoon garlic powder', '(makes about 4 cups)', 'Preheat the oven to 350 F. Butter or oil an 8-inch baking dish.',
    #         'Cook the penne 2 minutes less than package directions.', '(It will finish cooking in the oven.)', 'Rinse the pasta in cold water and set aside.',
    #         'Combine the cooked pasta and the sauce in a medium bowl and mix carefully but thoroughly.', 'Scrape the pasta into the prepared baking dish.',
    #         'Sprinkle the top with the cheeses and then the chili powder.', 'Bake, uncovered, for 20 minutes.', 'Let the mac and cheese sit for 5 minutes before serving.',
    #         'Melt the butter in a heavy-bottomed saucepan over medium heat and whisk in the flour.', 'Continue whisking and cooking for 2 minutes.',
    #         'Slowly add the milk, whisking constantly.', 'Cook until the sauce thickens, about 10 minutes, stirring frequently.', 'Remove from the heat.',
    #         'Add the cheeses, salt, chili powder, and garlic powder.', 'Stir until the cheese is melted and all ingredients are incorporated, about 3 minutes.',
    #         'Use immediately, or refrigerate for up to 3 days.', 'This sauce reheats nicely on the stove in a saucepan over low heat.',
    #         'Stir frequently so the sauce doesnt scorch.',
    #         'This recipe can be assembled before baking and frozen for up to 3 monthsjust be sure to use a freezer-to-oven pan and increase the baking time to 50 minutes.',
    #         'One-half teaspoon of chipotle chili powder makes a spicy mac, so make sure your family and friends can handle it!',
    #         'The proportion of pasta to cheese sauce is crucial to the success of the dish.',
    #         'It will look like a lot of sauce for the pasta, but some of the liquid will be absorbed.']
    # text2emb(text, model, tokenizer)


def text2emb(text, model, tokenizer, id, device):
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
        print(id, input_ids.shape)
        return []
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


