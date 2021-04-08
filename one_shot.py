import torch
import clip
from tqdm import tqdm
from data import PokemonDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

print(device)
dataset = PokemonDataset()
classes = list(dataset.get_classes())

proper_one = 0
proper_two = 0
all_one = 0
all_two = 0
for image, real_labels in tqdm(dataset):
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    text = clip.tokenize(classes).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        text_features = model.encode_text(text)

        logits_per_image, logits_per_text = model(image_tensor, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    if real_labels[1] is None:
        all_one += 1
        if real_labels[0] == probs.argmax:
            proper_one += 1
    else:
        argsorted_probs = probs[0].argsort()
        most_probable_indexes = [argsorted_probs[-1], argsorted_probs[-2]]
        all_two += 2
        if real_labels[0] in most_probable_indexes:
            proper_two += 1
        if real_labels[1] in most_probable_indexes:
            proper_two += 1

accuracy_one = proper_one/all_one
accuracy_two = proper_two/all_two
accuracy_all = (proper_one+proper_two) / (all_one+all_two)
print(f"One type pokemon accuracy: {accuracy_one}")
print(f"Two type pokemon accuracy: {accuracy_two}")
print(f"All pokemon accuracy: {accuracy_all}")
