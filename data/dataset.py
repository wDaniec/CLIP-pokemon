import os
import re
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


def prep():
    pokemons = pd.read_csv('./data/pokedex.csv')
    numbers = []
    for i in range(1, len(pokemons) + 1):
        numbers.append(i)
    pokemons['pkn'] = numbers
    img_dir = './data/images'
    onlyfiles = os.listdir(img_dir)

    dataframe_img = pd.DataFrame([])
    images = []
    pokemon_number = []
    for img in onlyfiles:
        if re.search('-', img):
            continue
        pkn = img[:-4]
        n = re.sub("[^0-9]", "", pkn)
        path = os.path.join(img_dir, img)
        images.append(path)
        pokemon_number.append(n)
    dataframe_img['path'] = images
    dataframe_img['pkn'] = pokemon_number
    dataframe_img['pkn'] = dataframe_img['pkn'].astype(int)
    result = pokemons.merge(dataframe_img, left_on='pkn', right_on='pkn')
    return result


class PokemonDataset(Dataset):
    def __init__(self, transform=None):
        super().__init__()
        self.img_labels = prep()
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        pokemon = self.img_labels.iloc[idx]
        image = Image.open(pokemon["path"])
        if self.transform:
            image = self.transform(image)
        label1 = pokemon["Type1"]
        label2 = pokemon["Type2"]
        label2 = None if pd.isnull(label2) else label2
        return image, [label1, label2]

    def get_classes(self):
        type1 = pd.unique(self.img_labels["Type1"])
        type2 = [x for x in pd.unique(self.img_labels["Type2"]) if not pd.isnull(x)]
        return set(type1) | set(type2)

    # to be used for few-shot learning regime; return n examples of each type
    # note: for training purposes we would like to pass each example with only one label
    def fetch_per_type_examples(self, n_examples=1):
        examples = []
        for cls in self.get_classes():
            cur = self.img_labels.query(f'"{cls}" in Type1 or "{cls}" in Type2').sample(n_examples)
            for i in range(n_examples):
                pokemon = cur.iloc[i]
                image = Image.open(pokemon["path"])
                if self.transform:
                    image = self.transform(image)
                examples.append((image, cls))
        return examples
