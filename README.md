# CLIP-pokemon
Mini-project#1 for class "Deep Learning with Multiple Tasks". The project goal is to create dataset of pokemons and use pretrained CLIP model to classify what type of the pokemon is. 


1. Dataset should be downloaded from https://www.kaggle.com/kvpratama/pokemon-images-dataset and put into folder
data/images

2. Pytorch should be installed:
```shell
conda install --yes -c pytorch pytorch=1.7.1 torchvision cpuonly
pip install git+https://github.com/openai/CLIP.git
```

---
Little insight into the Pokemon world:

![dataset_preview](./figures_and_metrics/dataset_preview.png?raw=true)

https://github.com/wDaniec/CLIP-pokemon/tree/main
As we see above, each of them belongs to one or two types (classes), so the problem CLIP will be facing is kind of multilabel classification.

Some gist of numbers in the dataset:
- number of Pokemons:  707 (of one-type: 345 and two-type: 362)
- number of types: 18

We want to check whether CLIP is able of distinguishing between these clases.
The task is really hard but could potentially give us some insight, what CLIP is capable of;
some of our Pocket Monsters have  

Let's focus on distributions between classes:
