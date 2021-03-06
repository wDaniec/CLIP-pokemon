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

        # cls_preds = pd.concat((preds[cls].sort_values()[-5:][::-1], preds[cls].sort_values()[:5]))
        cls_preds = preds[cls].sort_values()[-10:][::-1]
        fig, axs = plt.subplots(2, 5, figsize=(10, 5))

        for (idx, prob), ax in zip(cls_preds.iteritems(), axs.flatten()):
            img, labels = dataset[idx]
            labels = ', '.join([l for l in labels if l])

            ax.imshow(img)
            ax.set_title(str(round(prob, 4)) + '\n' + labels)
            ax.axis('off')

        fig.tight_layout()
        plt.savefig(os.path.join("figs", f"{cls}_preds_preview.png"))


def peek_at_pokemons():
    pkm_idxs, cls = [170, 256, 53, 86, 115], "Water"
    # pkm_idxs, cls = [387, 125, 76, 145], "Fire"
    dataset = PokemonDataset()

    fig, axs = plt.subplots(2, len(pkm_idxs), figsize=(14, 6))

    for idx, pkm_idx in enumerate(pkm_idxs):
        pkm_img, pkm_labs = dataset[pkm_idx]
        pkm_labs = ', '.join([l for l in pkm_labs if l])
        pred = pd.read_csv(PREDS_PATH).iloc[pkm_idx]
        pred_clses = pred.keys()[1:]
        pred_vals = pred.to_numpy()[1:]

        axs[0, idx].imshow(pkm_img)
        axs[0, idx].set_title(pkm_labs)
        axs[0, idx].axis('off')

        axs[1, idx].barh(pred_clses, pred_vals, align='center')
        axs[1, idx].set_yticks(np.arange(len(pred_clses)))
        axs[1, idx].set_yticklabels(pred_clses)

    fig.tight_layout()
    plt.savefig(os.path.join("figs", f"{cls}_pokemons_preview.png"))


peek_at_pokemons()
