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

![dataset_preview](figs/dataset_preview.png?raw=true)

As we see above, each of them belongs to one or two types (classes), so the problem CLIP will be facing is kind of multilabel classification.

Some gist of numbers in the dataset:
- number of Pokemons:  708 (of one-type: 346 and of two-type: 362)
- number of types: 18 (Bug: 64, Dark: 44, Dragon: 37, Electric: 41, Fairy: 35, Fighting: 43, Fire: 56, Flying: 88, Ghost: 36, Grass: 79, Ground: 58, Ice: 33, Normal: 96, Poison: 59, Psychic: 73, Rock: 55, Steel: 41, Water: 116)

We want to check whether CLIP is able of distinguishing between these classes.
The task is really hard but could potentially give us some insight, what CLIP is capable of;
some of our Pocket Monsters have  subtle features which could give a human hint of its belonging.

Random chance of guessing is around 9%  while CLIP achieves 25%, what we calculated as average of accuracies among each case of one (20,2%) and two-types (27,6%) pocket monsters.

![](figs/accuracy.png)

We checked what CLIP perceives as most and least representative for each type. Additionaly we took a closer look at specific Pokemons odds. 
Here is an example for grass type:
![](figs/Grass_preds_preview.png?raw=true)
It looks like it have solid understanding what these concepts are as it produces reasonable and confident predictions.

For the fire type, situation is a bit different. Although it recognizes well, it is significantly less confident what we see as strange because of obvious flame marks.
![](figs/Fire_preds_preview.png?raw=true)
![](figs/Fire_pokemons_preview.png?raw=true)

With the water type CLIP failed to associate any Pokemon:
![](figs/Water_preds_preview.png?raw=true)
![](figs/Water_pokemons_preview.png?raw=true)

Overall we grade CLIP for a 3+ taking into account the task was hard.
