import operator
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import clip
from data import PokemonDataset

PREDS_PATH = os.path.join("data", "preds.csv")
POKEDEX_PATH = os.path.join("data", "pokedex.csv")


def clean_pokedex():
    pms = pd.DataFrame([[type1, type2] for _, [type1, type2] in PokemonDataset()], columns=["Type1", "Type2"])
    pms.to_csv(POKEDEX_PATH)


def dataset_preview():
    data = PokemonDataset().fetch_per_type_examples()
    fig, axs = plt.subplots(3, 6, figsize=(14, 7))

    for datum, ax in zip(data, axs.flatten()):
        img, cls = datum
        ax.imshow(img)
        ax.set_title(cls[0] if type(cls[1]) is float else f"{cls[0]}, {cls[1]}")
        ax.axis('off')

    fig.tight_layout()
    plt.savefig(os.path.join("figs", "dataset_preview.png"))


def count_type_reprs():
    pms = pd.read_csv(POKEDEX_PATH)
    print(pms['Type1'].value_counts() + pms['Type2'].value_counts())


def count_double_types():
    pms = pd.read_csv(POKEDEX_PATH)
    print(pms.count())


def peek_at_classes():
    dataset = PokemonDataset()
    classes = dataset.get_classes()
    preds = pd.read_csv(PREDS_PATH)

    for cls in classes:
        cls_preds = pd.concat((preds[cls].sort_values()[-5:][::-1], preds[cls].sort_values()[:5]))
        fig, axs = plt.subplots(2, 5, figsize=(10, 5))

        for (idx, prob), ax in zip(cls_preds.iteritems(), axs.flatten()):
            img, labels = dataset[idx]
            labels = ', '.join([l for l in labels if l])

            ax.imshow(img)
            ax.set_title(str(round(prob, 4)) + '\n' + labels)
            ax.axis('off')

        fig.tight_layout()
        plt.savefig(os.path.join("figs", f"{cls}_preds_preview.png"))


# clean_pokedex()
# dataset_preview()
# print(metrics)
# count_type_reprs()
# count_single_types()
peek_at_classes()
