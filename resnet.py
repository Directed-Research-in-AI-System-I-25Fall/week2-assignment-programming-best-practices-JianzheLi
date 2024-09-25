from transformers import AutoImageProcessor, ResNetModel
from datasets import load_dataset
import torch
from torchvision import transforms
from tqdm import tqdm

dataset = load_dataset("mnist")
transform = transforms.Compose([
    transforms.Grayscale(3),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = ResNetModel.from_pretrained("microsoft/resnet-50")
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for example in tqdm(dataset['test']):
        image = example['image']
        label = example['label']
        
        image = transform(image)
        inputs = image_processor(image, return_tensors="pt")
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state

        predicted = last_hidden_states.mean(dim=1).argmax(dim=1)
        correct += (predicted.squeeze() == label).sum().item()
        total += 1

accuracy = correct / total
print(accuracy)
