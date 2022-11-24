import pandas as pd
import sys
import torchvision

transform = torchvision.transforms.ToTensor()
BATCH_SIZE = 32

name = sys.argv[1]
path = sys.argv[2]

def mnist():
    df = pd.read_csv(path, sep=',', header=None)
    print(df)
    data = df[df.columns[1:]]
    labels = df[df.columns[0]]
    data.to_csv(name+'_data', index=False, header=False)
    labels.to_csv(name+'_labels', index=False, header=False)


def voicehd():
    df = pd.read_csv(path, sep=', ', header=None)
    data = df[df.columns[:-1]]
    labels = df[df.columns[-1]]
    data.to_csv(name+'_data', index=False, header=False)
    labels.to_csv(name+'_labels', index=False, header=False)

mnist()
