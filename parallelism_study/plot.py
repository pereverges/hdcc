import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
folder = 'results/'

file = folder + 'par_rasp_vector_size_hdcc_17_03_2023_16:27:10'
#file_baseline = folder + 'seq_opengpu_vector_size_hdcc_22_02_2023_15:21:54'
threads = 4
df = pd.read_csv(file)
#df_baseline = pd.read_csv(file_baseline)
inits = []
leng = len(df['Application'].unique())
label_seq = []
label_paral = []
label_value = []
c = 0

fig, axs = plt.subplots(1,2)

for i in df['Application'].unique():
    label_seq.append(i+' sequential')
    label_paral.append(i+' theoretical paralelization')
    label_value.append(i+' actual paralelization')
    inits.append(df[df['Application'] == i].loc[df['Threads'] == 1]['Time'])
    #plt.plot(range(1,threads+1),[df_baseline[df_baseline['Application'] == i]['Time']]*threads, '--', label=label_seq[c])
    axs[0].bar(df[df['Application'] == i]['Threads'], df[df['Application'] == i]['Time'], label=label_value[c])
    c += 1
perfect_parallelism = [[] for x in range(leng)]
proxs = [[] for x in range(leng)]
import matplotlib as mpl
mpl.style.use('default')

for i in range(threads):
    for j in range(leng):
        proxs = inits[j]/(i+1)
        perfect_parallelism[j].append(proxs.item())
color = ['#0f3b59','#c05f00','#2CA02C']

for j in range(leng):
    axs[0].plot(range(1,threads+1), perfect_parallelism[j], color[j], label=label_paral[j])
#plt.plot(range(1,threads+1), perfect_parallelism2, 'g--')


plt.title('Code parallelization ThunderX')






file = folder + 'thunder_par'
#file_baseline = folder + 'seq_opengpu_vector_size_hdcc_22_02_2023_15:21:54'
threads = 96
df = pd.read_csv(file)
#df_baseline = pd.read_csv(file_baseline)
inits = []
leng = len(df['Application'].unique())
label_seq = []
label_paral = []
label_value = []
c = 0


for i in df['Application'].unique():
    label_seq.append(i+' sequential')
    label_paral.append(i+' theoretical paralelization')
    label_value.append(i+' actual paralelization')
    inits.append(df[df['Application'] == i].loc[df['Threads'] == 1]['Time'])
    #plt.plot(range(1,threads+1),[df_baseline[df_baseline['Application'] == i]['Time']]*threads, '--', label=label_seq[c])
    axs[1].bar(df[df['Application'] == i]['Threads'], df[df['Application'] == i]['Time'], label=label_value[c])
    c += 1
perfect_parallelism = [[] for x in range(leng)]
proxs = [[] for x in range(leng)]
import matplotlib as mpl
mpl.style.use('default')

for i in range(threads):
    for j in range(leng):
        proxs = inits[j]/(i+1)
        perfect_parallelism[j].append(proxs.item())
color = ['#0f3b59','#c05f00','#2CA02C']

for j in range(leng):
    axs[1].plot(range(1,threads+1), perfect_parallelism[j], color[j], label=label_paral[j])
#plt.plot(range(1,threads+1), perfect_parallelism2, 'g--')










axs[0].set_title(label='Code parallelization RaspberryPi')
axs[0].set_ylabel('Time (s)')
axs[0].set_xlabel('Threads')

axs[1].set_title(label='Code parallelization ThunderX')
axs[1].set_ylabel('Time (s)')
axs[1].set_xlabel('Threads')
plt.legend()

plt.show()
