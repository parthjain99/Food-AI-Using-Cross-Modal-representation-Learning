# cs536_final_project
Detailed Information
https://github.com/parthjain99/Food-AI-Using-Cross-Modal-representation-Learning/blob/main/Final_report.pdf

# Description:


## model1

* model1/vit_model.py: extract image feature by ViT, save as a dict pickle file

* model/self_attn.py:  model1's architecture, training code, save model

* model/m1_evaluate.py: evaluate model1 by medR and Recall@K, K=1,5,10


## model2

* model2/demo.py: connect id--recipeId--image_path which is a sample, save as a dict

* model2/image_attn.py:  model2's training file, save model

* model2/evaluate.py: evaluate model2 by medR and Recall@K,  K=1,5,10

* model2/text_process.py: extract fine-grained text features(title, ingredients, instructions)

* model2/text_total_emb.py: extract image and recipe ground truth embedding

* model2/triplet.py: training model with triplet loss
