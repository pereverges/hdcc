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

machines_names = ['MAC','OPENLAB','OPENGPU']
dataset_hdcc = [pd.read_csv(file_hdcc), pd.read_csv(file_hdcc_openlab), pd.read_csv(file_hdcc_opengpu)]
dataset_torchhd = [pd.read_csv(file_torchhd), pd.read_csv(file_torchhd_openlab), pd.read_csv(file_torchhd_opengpu)]
dataset_hdcc_names = [dataset_hdcc[0].Application.unique(), dataset_hdcc[0].Application.unique(),
                      dataset_hdcc[0].Application.unique()]
first = True
machines = 3

for index, key in enumerate(dataset_hdcc_names[0]):
    fig, axs = plt.subplots(3, machines)
    for i in range(machines):
        print(dataset_hdcc[i].Application)
        print(key)
        time_hdc = dataset_hdcc[i][:][dataset_hdcc[i].Application == key][["Dimensions", "Application", "Time"]]
        time_hdc['Tool'] = 'HDCC'
        time_torchhd = dataset_torchhd[i][:][dataset_torchhd[i].Application == key][["Dimensions", "Application", "Time"]]
        time_torchhd['Tool'] = 'TORCHHD'

        accuracy_hdc = dataset_hdcc[i][:][dataset_hdcc[i].Application == key][["Dimensions", "Application", "Accuracy"]]
        accuracy_hdc['Tool'] = 'HDCC'
        accuracy_torchhd = dataset_torchhd[i][:][dataset_torchhd[i].Application == key][
            ["Dimensions", "Application", "Accuracy"]]
        accuracy_torchhd['Tool'] = 'TORCHHD'

        memory_hdc = dataset_hdcc[i][:][dataset_hdcc[i].Application == key][["Dimensions", "Application", "Memory"]]
        memory_hdc['Tool'] = 'HDCC'
        memory_torchhd = dataset_torchhd[i][:][dataset_torchhd[i].Application == key][
            ["Dimensions", "Application", "Memory"]]
        memory_torchhd['Tool'] = 'TORCHHD'
        # axs[1,i].set_ylim(ymin=0.6)
        axs[0, i].plot(time_hdc['Dimensions'], time_hdc['Time'], label='HDCC')
        axs[0, i].plot(time_torchhd['Dimensions'], time_torchhd['Time'], label='TORCHHD')
        axs[1, i].plot(accuracy_hdc['Dimensions'], accuracy_hdc['Accuracy'], label='HDCC')
        axs[1, i].plot(accuracy_torchhd['Dimensions'], accuracy_torchhd['Accuracy'], label='TORCHHD')
        axs[2, i].plot(accuracy_hdc['Dimensions'], memory_hdc['Memory'], label='HDCC')
        axs[2, i].plot(accuracy_torchhd['Dimensions'], memory_torchhd['Memory'], label='TORCHHD')

        axs[0, i].legend()
        axs[0, i].set_title(label='Time'+machines_names[i])
        axs[1, i].legend()
        axs[1, i].set_title(label='Accuracy')
        axs[2, i].legend()
        axs[2, i].set_title(label='Memory')
        fig.suptitle(key)
    plt.show()

