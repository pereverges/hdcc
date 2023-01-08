import pandas as pd
import matplotlib.pyplot as plt

folder = 'experiments/'
file_hdcc = folder + 'mac_num_threads_4_vector_size_hdcc_08_01_2023_11:10:42'
file_torchhd = folder + 'mac_torchhd_08_01_2023_10:50:10'

dataset_hdcc = pd.read_csv(file_hdcc)
dataset_torchhd = pd.read_csv(file_torchhd)
dataset_hdcc_names = dataset_hdcc.Application.unique()
first = True

for index, key in enumerate(dataset_hdcc_names):
    fig, axs = plt.subplots(2)

    time_hdc = dataset_hdcc[:][dataset_hdcc.Application == key][["Dimensions","Application","Time"]]
    time_hdc['Tool'] = 'HDCC'
    time_torchhd = dataset_torchhd[:][dataset_torchhd.Application == key][["Dimensions","Application","Time"]]
    time_torchhd['Tool'] = 'TORCHHD'

    accuracy_hdc = dataset_hdcc[:][dataset_hdcc.Application == key][["Dimensions","Application","Accuracy"]]
    accuracy_hdc['Tool'] = 'HDCC'
    accuracy_torchhd = dataset_torchhd[:][dataset_torchhd.Application == key][["Dimensions","Application","Accuracy"]]
    accuracy_torchhd['Tool'] = 'TORCHHD'
    axs[0].plot(time_hdc['Dimensions'], time_hdc['Time'], label='HDCC')
    axs[0].plot(time_torchhd['Dimensions'], time_torchhd['Time'], label='TORCHHD')
    axs[1].plot(accuracy_hdc['Dimensions'], accuracy_hdc['Accuracy'], label='HDCC')
    axs[1].plot(accuracy_torchhd['Dimensions'], accuracy_torchhd['Accuracy'], label='TORCHHD')
    axs[0].legend()
    axs[0].set_title(label='Time')
    axs[1].legend()
    axs[1].set_title(label='Accuracy')
    fig.suptitle(key)
    plt.show()

