import pandas as pd
import matplotlib.pyplot as plt

time = pd.read_csv('plot_data/exec_time.csv')
time = time.groupby('dimensions').mean().plot(title="Execution time", xlabel='Dimensions', ylabel='Time(s)')

memory = pd.read_csv('plot_data/memory.csv')
memory = memory.groupby('dimensions').mean().plot(title="Memory", xlabel='Dimensions', ylabel='Memory (kb)')

accuracy = pd.read_csv('plot_data/accuracy.csv')
accuracy = accuracy.groupby('dimensions').mean().plot(title="Accuracy", xlabel='Dimensions', ylabel='Accuracy (%)')

plt.legend()
plt.show()
