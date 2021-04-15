import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import clip
from data import PokemonDataset


def class_examples(path):
    single_class = False
    data = PokemonDataset().fetch_per_type_examples()
    fig, axs = plt.subplots(3, 6, figsize=(14, 7))

    for datum, ax in zip(data, axs.flatten()):
        img, cls = datum
        ax.imshow(img)
        ax.set_title(cls if single_class else (cls[0] if type(cls[1]) is float else f"{cls[0]}, {cls[1]}"))
        ax.axis('off')

    fig.tight_layout()
    plt.savefig(path)


def count_type_reprs(path):
    metrics = pd.read_csv(path)
    metrics.count(axis=1)


def count_single_types(path):
    metrics = pd.read_csv(path)
    metrics['MISSING'] = metrics.apply(lambda x: x.isnull().sum(), axis='columns')
    print(metrics['MISSING'])


def peek_at_classes(path):
    classes = list(PokemonDataset().get_classes())
    metrics = pd.read_csv(path)
    for cls in classes:
        cur_type = metrics.loc[metrics["_idx"] == cls].sort_values(by=classes.index(cls) + 2, ascending=False, axis=0)
        print(cur_type)
        break


PATH = os.path.join("figs", "preds.csv")
metrics = pd.read_csv(PATH)
print(metrics)
