import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

# Note: this example requires the torchmetrics library: https://torchmetrics.readthedocs.io
import torchmetrics
from tqdm import tqdm

from torchhd import functional
from torchhd import embeddings
from torchhd.datasets import EMGHandGestures

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))

DIMENSIONS = 1000  # number of hypervector dimensions
NUM_LEVELS = 21
BATCH_SIZE = 1  # for GPUs with enough memory we can process multiple images at ones
WINDOW = 256
N_GRAM_SIZE = 4
DOWNSAMPLE = 5
SUBSAMPLES = torch.arange(0, WINDOW, int(WINDOW / DOWNSAMPLE))


def transform(x):
    return x[SUBSAMPLES]


class Model(nn.Module):
    def __init__(self, num_classes, timestamps, channels):
        super(Model, self).__init__()

        #self.channels = embeddings.Random(channels, DIMENSIONS)
        self.channels = embeddings.Random(1024, DIMENSIONS)
        #self.timestamps = embeddings.Random(timestamps, DIMENSIONS)
        self.timestamps = embeddings.Random(1, DIMENSIONS)
        self.signals = embeddings.Level(NUM_LEVELS, DIMENSIONS, high=20)
        self.flatten = torch.nn.Flatten()

        self.classify = nn.Linear(DIMENSIONS, num_classes, bias=False)
        self.classify.weight.data.fill_(0.0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        #print(x.shape)

        signal = self.signals(x)
        #print(signal.shape)
        samples = functional.bind(signal, self.channels.weight.unsqueeze(0))
        #samples = functional.bind(samples, self.timestamps.weight.unsqueeze(1))

        samples = functional.multiset(samples)
        #samples = functional.permute(samples, shifts=N_GRAM_SIZE)
        #samples = functional.ngrams(samples, n=N_GRAM_SIZE)
        return functional.hard_quantize(samples)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc = self.encode(x)
        logit = self.classify(enc)
        return logit


def experiment(subjects=[0]):
    print("List of subjects " + str(subjects))
    ds = EMGHandGestures(
        "../data", download=True, subjects=subjects
    )

    train_size = int(len(ds) * 0.7)
    test_size = len(ds) - train_size
    train_ds, test_ds = data.random_split(ds, [train_size, test_size])

    train_ld = data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_ld = data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = Model(len(ds.classes), ds[0][0].size(-2), ds[0][0].size(-1))
    model = model.to(device)

    with torch.no_grad():
        for index, (samples, labels) in enumerate(tqdm(train_ld, desc="Training")):
            if index == 0:
                print(samples.shape)
            samples = samples.to(device)
            labels = labels.to(device)

            samples_hv = model.encode(samples)
            model.classify.weight[labels] += samples_hv
            #print(labels)

        model.classify.weight[:] = F.normalize(model.classify.weight)

    # accuracy = torchmetrics.Accuracy()
    suma = 0
    with torch.no_grad():
        for samples, labels in test_ld:
            samples = samples.to(device)

            outputs = model(samples)
            predictions = torch.argmax(outputs, dim=-1)
            if predictions == labels[0]:
                # print(predictions)
                suma += 1
            #accuracy.update(predictions.cpu(), labels)
    print((suma/len(test_ld))*100, suma, len(test_ld))
    #print(f"Testing accuracy of {(accuracy.compute().item() * 100):.3f}%")


# Make a model for each subject
experiment([0])
'''
for i in range(5):
    experiment([i])
'''