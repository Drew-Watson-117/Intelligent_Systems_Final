import matplotlib.pyplot as plt
import numpy as np


data = []
dataCount = 10000

for i in range(dataCount):
    data.append(np.arange(i,i+5))

for i in range(len(data)):
    plt.plot(i,data[i][0],'r.')
plt.semilogy()
plt.show()