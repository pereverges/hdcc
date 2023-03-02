import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
folder = 'results/'

file = folder + 'opengpu_vector_size_hdcc_22_02_2023_11:40:56'
file_baseline = folder + 'seq_opengpu_vector_size_hdcc_22_02_2023_15:21:54'
threads = 39
df = pd.read_csv(file)
df_baseline = pd.read_csv(file_baseline)
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
    plt.plot(range(1,threads+1),[df_baseline[df_baseline['Application'] == i]['Time']]*threads, '--', label=label_seq[c])
    plt.bar(df[df['Application'] == i]['Threads'], df[df['Application'] == i]['Time'], label=label_value[c])
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
    plt.plot(range(1,threads+1), perfect_parallelism[j], color[j], label=label_paral[j])
#plt.plot(range(1,threads+1), perfect_parallelism2, 'g--')
plt.legend()
plt.xlabel('Threads')
plt.ylabel('Time')
plt.title('Code paralelization')
plt.show()
