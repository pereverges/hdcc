import pandas as pd
import matplotlib.pyplot as plt

folder = 'experiments/'
# file_hdcc = folder + 'opengpu_num_threads_20_vector_size_hdcc_08_01_2023_00:03:30'
file_hdcc = folder + 'mac_num_threads_4_vector_size_hdcc_13_01_2023_07:49:22'
file_torchhd = folder + 'mac_torchhd_13_01_2023_13:37:40'

file_torchhd_opengpu = folder + 'opengpu_torchhd_13_01_2023_15:50:42'
file_hdcc_opengpu = folder + 'opengpu_num_threads_20_vector_size_hdcc_13_01_2023_15:02:23'

file_torchhd_openlab = folder + 'openlab_torchhd_13_01_2023_17:30:49'
file_hdcc_openlab = folder + 'openlab_num_threads_20_vector_size_hdcc_13_01_2023_15:42:08'
# file_torchhd = folder + 'opengpu_torchhd_08_01_2023_07:21:50'

machines_names = ['ARM','Intel1','Intel2']
dataset_hdcc = [pd.read_csv(file_hdcc), pd.read_csv(file_hdcc_openlab), pd.read_csv(file_hdcc_opengpu)]
dataset_torchhd = [pd.read_csv(file_torchhd), pd.read_csv(file_torchhd_openlab), pd.read_csv(file_torchhd_opengpu)]
dataset_hdcc_names = [dataset_hdcc[0].Application.unique(), dataset_hdcc[0].Application.unique(),
                      dataset_hdcc[0].Application.unique()]
first = True
machines = 3
names = ['ISOLET','MNIST','EMG','LANGUAGE']
fig, axs = plt.subplots(3, 4)
key = ['voicehd','mnist','emgppp','languages']
for j in range(len(key)):
    for i in range(machines):
        time_hdc = dataset_hdcc[i][:][dataset_hdcc[i].Application == key[j]][["Dimensions", "Application", "Time"]]
        time_hdc['Tool'] = 'HDCC'
        time_torchhd = dataset_torchhd[i][:][dataset_torchhd[i].Application == key[j]][["Dimensions", "Application", "Time"]]
        time_torchhd['Tool'] = 'TORCHHD'

        accuracy_hdc = dataset_hdcc[i][:][dataset_hdcc[i].Application == key[j]][["Dimensions", "Application", "Accuracy"]]
        accuracy_hdc['Tool'] = 'HDCC'
        accuracy_torchhd = dataset_torchhd[i][:][dataset_torchhd[i].Application == key[j]][
            ["Dimensions", "Application", "Accuracy"]]
        accuracy_torchhd['Tool'] = 'TORCHHD'

        memory_hdc = dataset_hdcc[i][:][dataset_hdcc[i].Application == key[j]][["Dimensions", "Application", "Memory"]]
        memory_hdc['Tool'] = 'HDCC'
        memory_torchhd = dataset_torchhd[i][:][dataset_torchhd[i].Application == key[j]][
            ["Dimensions", "Application", "Memory"]]
        memory_torchhd['Tool'] = 'TORCHHD'
        # axs[1,i].set_ylim(ymin=0.6)
        axs[0, j].plot(time_hdc['Dimensions'], time_hdc['Time'], label='HDCC '+machines_names[i])
        axs[0, j].plot(time_torchhd['Dimensions'], time_torchhd['Time'], label='TorchHD '+machines_names[i])
        axs[1, j].plot(accuracy_hdc['Dimensions'], accuracy_hdc['Accuracy'], label='HDCC '+machines_names[i])
        axs[1, j].plot(accuracy_torchhd['Dimensions'], accuracy_torchhd['Accuracy'], label='TorchHD '+machines_names[i])
        axs[2, j].plot(accuracy_hdc['Dimensions'], memory_hdc['Memory'], label='HDCC '+machines_names[i])
        axs[2, j].plot(accuracy_torchhd['Dimensions'], memory_torchhd['Memory'], label='TorchHD '+machines_names[i])

    axs[0, j].set_title(label='Time '+names[j])
    axs[0, j].set_xlabel('Dimensions')
    axs[0, j].set_ylabel('Time (s)')

    axs[1, j].set_title(label='Accuracy '+names[j])
    axs[1, j].set_xlabel('Dimensions')
    axs[1, j].set_ylabel('Accuracy')

    axs[2, j].set_title(label='Peak memory '+names[j])
    axs[2, j].set_xlabel('Dimensions')
    axs[2, j].set_ylabel('Memory (bytes)')


plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
