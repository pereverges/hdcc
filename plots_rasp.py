import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
folder = 'results/'
# file_hdcc = folder + 'opengpu_num_threads_20_vector_size_hdcc_08_01_2023_00:03:30'
file_hdcc = folder + 'pi_num_threads_4_vector_size_hdcc_03_02_2023_16:19:00'
file_torchhd = folder + 'rasp_torchhd_11_02_2023_15:44:11'

machines_names = ['Rasp']
dataset_hdcc = [pd.read_csv(file_hdcc)]
dataset_torchhd = [pd.read_csv(file_torchhd)]

first = True
machines = 1
names = ['Isolet','Mnist','Emg','Lang recognition']
fig, axs = plt.subplots(3,4)
key = ['voicehd','mnist','emgppp','languages']
for j in range(len(key)):
    for i in range(machines):
        time_hdc = dataset_hdcc[i][:][dataset_hdcc[i].Application == key[j]][["Dimensions", "Application", "Time"]].groupby('Dimensions').mean().reset_index()
        time_hdc['Tool'] = 'HDCC'
        time_torchhd = dataset_torchhd[i][:][dataset_torchhd[i].Application == key[j]][["Dimensions", "Application", "Time"]].groupby('Dimensions').mean().reset_index()
        time_torchhd['Tool'] = 'TORCHHD'
        accuracy_hdc = dataset_hdcc[i][:][dataset_hdcc[i].Application == key[j]][["Dimensions", "Application", "Accuracy"]].groupby('Dimensions').mean().reset_index()
        accuracy_hdc['Tool'] = 'HDCC'
        accuracy_torchhd = dataset_torchhd[i][:][dataset_torchhd[i].Application == key[j]][
            ["Dimensions", "Application", "Accuracy"]].groupby('Dimensions').mean().reset_index()
        accuracy_torchhd['Tool'] = 'TORCHHD'

        memory_hdc = dataset_hdcc[i][:][dataset_hdcc[i].Application == key[j]][["Dimensions", "Application", "Memory"]].groupby('Dimensions').mean().reset_index()
        memory_hdc['Tool'] = 'HDCC'
        memory_torchhd = dataset_torchhd[i][:][dataset_torchhd[i].Application == key[j]][
            ["Dimensions", "Application", "Memory"]].groupby('Dimensions').mean().reset_index()


        memory_torchhd['Tool'] = 'TORCHHD'
        # axs[1,i].set_ylim(ymin=0.6)
        axs[0, j].plot(time_hdc['Dimensions'], time_hdc['Time'], label='HDCC')
        axs[0, j].plot(time_torchhd['Dimensions'], time_torchhd['Time'], label='TorchHD')
        axs[1, j].plot(accuracy_hdc['Dimensions'], accuracy_hdc['Accuracy'], label='HDCC')
        axs[1, j].plot(accuracy_torchhd['Dimensions'], accuracy_torchhd['Accuracy'], label='TorchHD')
        axs[2, j].plot(memory_hdc['Dimensions'], memory_hdc['Memory'], label='HDCC')
        axs[2, j].plot(memory_torchhd['Dimensions'], memory_torchhd['Memory'], label='TorchHD')

    axs[0, j].set_title(label=names[j]+'\n\nTime')
    axs[0, j].set_xlabel('Dimensions')
    axs[0, j].set_ylabel('Time (s)')

    axs[1, j].set_title(label='Accuracy')
    axs[1, j].set_xlabel('Dimensions')
    axs[1, j].set_ylabel('Accuracy')

    axs[2, j].set_title(label='Peak memory')
    axs[2, j].set_xlabel('Dimensions')
    axs[2, j].set_ylabel('Memory (bytes)')


plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
