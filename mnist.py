import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.datasets import MNIST
from torchhd import functional
from torchhd import embeddings
import torch
import time
import sys

BATCH_SIZE = 1
DIMENSIONS = int(sys.argv[1])
IMG_SIZE = 28
NUM_LEVELS = 1000
NUM_CLASSES = 10
transform = torchvision.transforms.ToTensor()

train_ds = MNIST("../data", train=True, transform=transform, download=True)
train_ds = torch.utils.data.Subset(train_ds, range(60000))

train_ld = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)

test_ds = MNIST("../data", train=False, transform=transform, download=True)
test_ds = torch.utils.data.Subset(test_ds, range(1000))
test_ld = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

position = embeddings.Random(784, DIMENSIONS)
value = embeddings.Level(NUM_LEVELS, DIMENSIONS)


flatten = torch.nn.Flatten()
classify = nn.Linear(DIMENSIONS, NUM_CLASSES, bias=False)
classify.weight.data.fill_(0.0)

r = flatten(value.weight)
x = flatten(position.weight)

def encoding(x):
    x = flatten(x)
    sample_hv = functional.bind(position.weight, value(x))
    sample_hv = functional.multiset(sample_hv)
    sample_hv = functional.hard_quantize(sample_hv)
    return sample_hv


def classification(x):
    enc = encoding(x)
    logit = classify(enc)
    return logit


t = time.time()
with torch.no_grad():
    for samples, labels in train_ld:
        samples_hv = encoding(samples)
        classify.weight[labels] += samples_hv
    classify.weight[:] = F.normalize(classify.weight)
correct_pred = 0

with torch.no_grad():
    for samples, labels in test_ld:
        outputs = classification(samples)
        predictions = torch.argmax(outputs, dim=-1)
        if predictions == labels:
            correct_pred += 1

#print('Time', time.time()-t)
#print('Accuracy', (correct_pred/len(labels)/10000))
print('mnist,'+str(DIMENSIONS) +',' + str(time.time()-t)+','+str((correct_pred/len(labels)/1000)))