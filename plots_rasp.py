import pandas as pd
import matplotlib.pyplot as plt
import warnings
import numpy as np
warnings.filterwarnings("ignore")
folder = 'results/'
# file_hdcc = folder + 'opengpu_num_threads_20_vector_size_hdcc_08_01_2023_00:03:30'
file_hdcc_our = folder + 'pi_execution_def'
file_hdcc = folder + 'pi_num_threads_4_vector_size_hdcc_03_02_2023_16:19:00'
file_torchhd = folder + 'rasp_torchhd_11_02_2023_15:44:11'

machines_names = ['Rasp']
dataset_hdcc = [pd.read_csv(file_hdcc)]
dataset_torchhd = [pd.read_csv(file_torchhd)]
dataset_our = [pd.read_csv(file_hdcc_our)]

first = True
machines = 1
names = ['Isolet','Mnist','Emg','Lang recognition']
fig, axs = plt.subplots(2,4)
key = ['voicehd','mnist','emgppp','languages']

x = np.arange(6)
width = 0.55


for j in range(len(key)):
    multiplier = 0

    for i in range(machines):
        offset = x+multiplier
        time_hdc_our = dataset_our[i][:][dataset_our[i].Application == key[j]][["Dimensions", "Application", "Time"]].groupby('Dimensions').mean().reset_index()
        time_hdc_our['Tool'] = 'OUR'

        time_hdc = dataset_hdcc[i][:][dataset_hdcc[i].Application == key[j]][["Dimensions", "Application", "Time"]].groupby('Dimensions').mean().reset_index()
        time_hdc['Tool'] = 'HDCC'

        time_torchhd = dataset_torchhd[i][:][dataset_torchhd[i].Application == key[j]][["Dimensions", "Application", "Time"]].groupby('Dimensions').mean().reset_index()
        time_torchhd['Tool'] = 'TORCHHD'

        accuracy_hdc_our = dataset_our[i][:][dataset_our[i].Application == key[j]][["Dimensions", "Application", "Accuracy"]].groupby('Dimensions').mean().reset_index()
        accuracy_hdc_our['Tool'] = 'OUR'

        accuracy_hdc = dataset_hdcc[i][:][dataset_hdcc[i].Application == key[j]][["Dimensions", "Application", "Accuracy"]].groupby('Dimensions').mean().reset_index()
        accuracy_hdc['Tool'] = 'HDCC'

        accuracy_torchhd = dataset_torchhd[i][:][dataset_torchhd[i].Application == key[j]][
            ["Dimensions", "Application", "Accuracy"]].groupby('Dimensions').mean().reset_index()
        accuracy_torchhd['Tool'] = 'TORCHHD'

        memory_hdc = dataset_hdcc[i][:][dataset_hdcc[i].Application == key[j]][["Dimensions", "Application", "Memory"]].groupby('Dimensions').mean().reset_index()
        memory_hdc['Tool'] = 'HDCC'

        memory_hdc_our = dataset_our[i][:][dataset_our[i].Application == key[j]][["Dimensions", "Application", "Memory"]].groupby('Dimensions').mean().reset_index()
        memory_hdc_our['Tool'] = 'OUR'

        memory_torchhd = dataset_torchhd[i][:][dataset_torchhd[i].Application == key[j]][
            ["Dimensions", "Application", "Memory"]].groupby('Dimensions').mean().reset_index()
        memory_torchhd['Tool'] = 'TORCHHD'

        # axs[1,i].set_ylim(ymin=0.6)
        axs[0, j].bar(x+offset+width,  time_hdc['Time'], width, label='HDCC')
        axs[0, j].bar(x+offset-width+width, time_torchhd['Time'], width, label='TORCHHD')
        axs[0, j].bar(x+offset+width+width, time_hdc_our['Time'], width, label='OUR')
        #axs[0, j].bar_label(r, padding=3)
        #axs[0, j].plot(time_hdc['Dimensions'], time_hdc['Time'], label='HDCC')
        #axs[0, j].plot(time_torchhd['Dimensions'], time_torchhd['Time'], label='TorchHD')
        #axs[1, j].plot(accuracy_hdc['Dimensions'], accuracy_hdc['Accuracy'], label='HDCC')
        #axs[1, j].plot(accuracy_torchhd['Dimensions'], accuracy_torchhd['Accuracy'], label='TorchHD')


        #axs[1, j].plot(memory_hdc['Dimensions'], memory_hdc['Memory'], label='HDCC')
        #axs[1, j].plot(memory_torchhd['Dimensions'], memory_torchhd['Memory'], label='TorchHD')

        #axs[1, j].plot(memory_hdc['Dimensions'], memory_hdc['Memory'], label='HDCC')
        #axs[1, j].plot(memory_torchhd['Dimensions'], memory_torchhd['Memory'], label='TORCHHD')
        axs[1, j].bar(x+offset+width,  memory_hdc['Memory'], width, label='HDCC')
        axs[1, j].bar(x+offset-width+width, memory_torchhd['Memory'], width, label='TORCHHD')
        axs[1, j].bar(x+offset+width+width, memory_hdc_our['Memory'], width, label='OUR')

        multiplier += 1


    axs[0, j].set_title(label=names[j]+'\n\nTime (s)')
    axs[0, j].set_xlabel('Dimensions')
    axs[0, j].set_ylabel('Time (s)')
    axs[0, j].set_xticks(x+offset+width, memory_hdc['Dimensions'])
    #axs[1, j].set_title(label='Accuracy')
    #axs[1, j].set_xlabel('Dimensions')
    #axs[1, j].set_ylabel('Accuracy')

    axs[1, j].set_title(label='Peak memory')
    axs[1, j].set_xlabel('Dimensions')
    axs[1, j].set_ylabel('Memory (bytes)')
    axs[1, j].set_xticks(x+offset+width, memory_hdc['Dimensions'])


plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
