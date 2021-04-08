import torch
import clip
from data import PokemonDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
optimizer.zero_grad()

dataset = PokemonDataset()
classes = list(dataset.get_classes())
text = clip.tokenize(classes).to(device)
examples = dataset.fetch_per_type_examples()

loss = 0
for ex in examples:
    image, label = ex
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    image_features = model.encode_image(image_tensor)
    text_features = model.encode_text(text)

    logits_per_image, logits_per_text = model(image_tensor, text)
    print(loss)
    loss += torch.nn.functional.cross_entropy(logits_per_image, label)

loss.backward()
optimizer.step()

# evaluate

# probs = logits_per_image.softmax(dim=-1).cpu().numpy()
# probs = logits_per_text.softmax(dim=-1).cpu().numpy()
