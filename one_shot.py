import torch
import clip
from data import PokemonDataset
from tqdm import tqdm

from collections import defaultdict

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.0005, momentum=0.5)

dataset = PokemonDataset()
classes = list(dataset.get_classes())
text = clip.tokenize(classes).to(device)
examples, pokemon_ids_to_omit = dataset.fetch_per_type_examples(5)
for _ in tqdm(range(75)):
    optimizer.zero_grad()

    labels = []
    images = []
    for ex in examples:
        image, label = ex
        labels.append(classes.index(label))
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        images.append(image_tensor)
    
    logits, _ = model(torch.cat(images), text)
    probs = logits.softmax(dim=-1)

    loss = torch.nn.functional.cross_entropy(probs, torch.as_tensor(labels).to(device))
    print(loss.cpu().item())
    loss.backward()
    optimizer.step()

# evaluation
proper_one = defaultdict(int)
proper_two = defaultdict(int)
all_one = defaultdict(int)
all_two = defaultdict(int)
text = clip.tokenize(classes).to(device)
print(pokemon_ids_to_omit)
for i, (image, real_labels) in enumerate(tqdm(dataset)):
    if i in pokemon_ids_to_omit:
        continue
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():

        logits_per_image, logits_per_text = model(image_tensor, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    if real_labels[1] is None:
        label = real_labels[0]
        all_one[label] += 1
        if real_labels[0] == classes[probs.argmax()]:
            proper_one[label] += 1
    else:
        argsorted_probs = probs[0].argsort()
        most_probable_indexes = [classes[argsorted_probs[-1]], classes[argsorted_probs[-2]]]
        all_two[real_labels[0]] += 1
        all_two[real_labels[1]] += 1
        if real_labels[0] in most_probable_indexes:
            proper_two[real_labels[0]] += 1
        if real_labels[1] in most_probable_indexes:
            proper_two[real_labels[1]] += 1

accuracy_one = sum(proper_one.values())/sum(all_one.values())
accuracy_two = sum(proper_two.values())/sum(all_two.values())
accuracy_all = (sum(proper_one.values())+sum(proper_two.values())) / (sum(all_one.values())+sum(all_two.values()))
print(f"One type pokemon accuracy: {accuracy_one}")
print(f"Two type pokemon accuracy: {accuracy_two}")
print(f"All pokemon accuracy: {accuracy_all}")

accuracy_per_class = {x:(proper_one[x] + proper_two[x])/(all_one[x] + all_two[x]) for x in classes}

print(accuracy_per_class)

