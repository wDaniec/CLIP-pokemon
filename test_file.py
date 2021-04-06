from data import PokemonDataset

dataset = PokemonDataset()
classes = dataset.get_classes()
for image, label in dataset:
    print(image, label)