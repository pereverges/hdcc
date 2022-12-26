import pandas as pd
import sys
import torchvision
from sklearn.utils import shuffle
import torch.utils.data as data
import torch
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
    print(name+'_data')


def emg_hand_data():
    path_labels = sys.argv[3]

    df_data = pd.read_csv(path, sep=',', header=None)
    df_labels = pd.read_csv(path_labels, sep=',', header=None)

    data = []
    labels = []
    iter = []

    for i in range(len(df_data)):
        if i % 256 == 0:
            labels.append(df_labels.iloc[i]-1)
            iter = []
            if i != 0:
                data.append(iter)
        iter += list(df_data.iloc[i])
    data.append(iter)
    train = int(len(data)*0.7)

    data, labels = shuffle(data, labels, random_state=0)

    df_train_data = pd.DataFrame(data[:train])
    df_train_data.to_csv(name+'_train_data', index=False, header=False)

    df_test_data = pd.DataFrame(data[train:])
    df_test_data.to_csv(name+'_test_data', index=False, header=False)

    df_train_labels = pd.DataFrame(labels[:train])
    df_train_labels.to_csv(name+'_train_labels', index=False, header=False)

    df_test_labels = pd.DataFrame(labels[train:])
    df_test_labels.to_csv(name+'_test_labels', index=False, header=False)

def emg_hand_labels():
    df = pd.read_csv(path, sep=',', header=None)
    data = []
    for i in range(len(df)):
        if i % 1024 == 0:
            data.append(df.iloc[i])
    train = int(len(data)*0.7)
    df_train = pd.DataFrame(data[:train])
    df_train.to_csv(name+'_train_labels', index=False, header=False)

    df_test = pd.DataFrame(data[train:])
    df_test.to_csv(name+'_test_labels', index=False, header=False)



def loadEMG():
    from torchhd.datasets import EMGHandGestures
    ds = EMGHandGestures(
        "../data", download=True, subjects=[1]
    )

    train_size = int(len(ds) * 0.7)
    test_size = len(ds) - train_size
    train_ds, test_ds = data.random_split(ds, [train_size, test_size])

    train_ld = data.DataLoader(train_ds, batch_size=1, shuffle=False)
    test_ld = data.DataLoader(test_ds, batch_size=1, shuffle=False)

    flatten = torch.nn.Flatten()

    train_data = []
    train_labels = []
    for samples, labels in train_ld:
        print(samples)
        train_data += flatten(samples).tolist()
        train_labels.append(labels.item())
    df_train_data = pd.DataFrame(train_data)
    df_train_data.to_csv(name + '_train_data', index=False, header=False)
    df_train_labels = pd.DataFrame(train_labels)
    df_train_labels.to_csv(name+'_train_labels', index=False, header=False)

    test_data = []
    test_labels = []
    for samples, labels in test_ld:
        test_data += flatten(samples).tolist()
        test_labels.append(labels.item())
    df_test_data = pd.DataFrame(test_data)
    df_test_data.to_csv(name + '_test_data', index=False, header=False)
    df_test_labels = pd.DataFrame(test_labels)
    df_test_labels.to_csv(name+'_test_labels', index=False, header=False)

loadEMG()
