import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import time
# Note: this example requires the torchmetrics library: https://torchmetrics.readthedocs.io
from tqdm import tqdm

import sys
from torchhd import functional
from torchhd import embeddings
from torchhd.datasets import EMGHandGestures
device = torch.device("cpu")
#print("Using {} device".format(device))

DIMENSIONS = int(sys.argv[2])
NUM_LEVELS = 21
BATCH_SIZE = 1  # for GPUs with enough memory we can process multiple images at ones
WINDOW = 256
N_GRAM_SIZE = 4
DOWNSAMPLE = 5
SUBSAMPLES = torch.arange(0, WINDOW, int(WINDOW / DOWNSAMPLE))

subjects=[int(sys.argv[1])]
def transform(x):
    return x[SUBSAMPLES]

class Model(nn.Module):
    def __init__(self, num_classes, timestamps, channels):
        super(Model, self).__init__()

        # self.channels = embeddings.Random(channels, DIMENSIONS)

        self.channels = embeddings.Random(1024, DIMENSIONS)
        # print(self.channels.weight.element_size()*self.channels.weight.nelement())
        # self.timestamps = embeddings.Random(timestamps, DIMENSIONS)
        #self.timestamps = embeddings.Random(1, DIMENSIONS)
        self.signals = embeddings.Level(NUM_LEVELS, DIMENSIONS, high=20)
        self.flatten = torch.nn.Flatten()

        self.classify = nn.Linear(DIMENSIONS, num_classes, bias=False)
        self.classify.weight.data.fill_(0.0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        # print(x.shape)

        signal = self.signals(x)
        samples = functional.bind(signal, self.channels.weight.unsqueeze(0))
        #samples = functional.bind(samples, self.timestamps.weight.unsqueeze(1))

        samples = functional.multiset(samples)
        samples = functional.permute(samples)

        return functional.hard_quantize(samples)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc = self.encode(x)
        logit = self.classify(enc)
        return logit

#("List of subjects " + str(subjects))
ds = EMGHandGestures(
    "../data", download=False, subjects=subjects
)

train_size = int(len(ds) * 0.7)
test_size = len(ds) - train_size
train_ds, test_ds = data.random_split(ds, [train_size, test_size])

train_ld = data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
test_ld = data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

model = Model(len(ds.classes), ds[0][0].size(-2), ds[0][0].size(-1))
model = model.to(device)
t = time.time()
with torch.no_grad():
    for samples, labels in tqdm(train_ld, desc="Training"):
        samples = samples.to(device)
        labels = labels.to(device)

        samples_hv = model.encode(samples)
        model.classify.weight[labels] += samples_hv

    model.classify.weight[:] = F.normalize(model.classify.weight)

# accuracy = torchmetrics.Accuracy()
suma = 0
with torch.no_grad():
    for samples, labels in tqdm(test_ld, desc="Testing"):
        samples = samples.to(device)

        outputs = model(samples)
        predictions = torch.argmax(outputs, dim=-1)
        #print(predictions, labels)
        if predictions == labels[0]:
            suma += 1
        #accuracy.update(predictions.cpu(), labels)
#print((suma/len(test_ld))*100)
#print(f"Testing accuracy of {(accuracy.compute().item() * 100):.3f}%")
n = ''
if subjects == [0]:
    n = 'emgp'
if subjects == [1]:
    n = 'emgpp'
if subjects == [2]:
    n = 'emgppp'
if subjects == [3]:
    n = 'emgpppp'
if subjects == [4]:
    n = 'emgppppp'
print(n+','+str(DIMENSIONS) +',' + str(time.time()-t)+','+str((suma/len(test_ld))), end = '')

# Make a model for each subject
'''
tracemalloc.start()
experiment([0])
print(',',tracemalloc.get_traced_memory()[1])
tracemalloc.reset_peak()
tracemalloc.stop()

tracemalloc.start()
experiment([1])
print(',',tracemalloc.get_traced_memory()[1])
tracemalloc.reset_peak()
tracemalloc.stop()

tracemalloc.start()
experiment([2])
print(',',tracemalloc.get_traced_memory()[1])
tracemalloc.reset_peak()
tracemalloc.stop()

tracemalloc.start()
experiment([3])
print(',',tracemalloc.get_traced_memory()[1])
tracemalloc.reset_peak()
tracemalloc.stop()
'''

