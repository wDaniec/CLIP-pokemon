import os
import time

import pandas as pd
import torch
import clip
from data import PokemonDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

dataset = PokemonDataset()
classes = list(dataset.get_classes())
print(len(classes))

stats = {'_idx': ['real_type_1', 'real_type_2', *classes]}

for i in range(len(dataset)):
    image, real_labels = dataset[i]
    stats[i] = [*real_labels]
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    text = clip.tokenize(classes).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        text_features = model.encode_text(text)

        logits_per_image, logits_per_text = model(image_tensor, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    # image.show()
    # print("Label probs:", [f"{clazz}: {x}" for x, clazz in zip(probs[0], classes)])
    # print("Real labels:", real_labels)
    # print("Most probable labels:", classes[probs[0].argmax()], classes[probs[0].argsort()[-2]])

    stats[i] += list(*probs)

df = pd.DataFrame(stats)
df.to_csv(os.path.join(os.environ['HOME'], "plots", "preds.csv"))
