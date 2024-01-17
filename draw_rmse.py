import matplotlib.pyplot as plt

import random

x_data = [i for i in range(3, 13, 3)]
# T-GCN
y_data = [4.149, 4.163, 4.179, 4.181]
# GRU
y2_data = [4.255, 4.255, 4.237, 4.223]
# ET-GCN
y3_data = [3.976, 3.996, 4.006, 4.020]

plt.rcParams["font.sans-serif"] = ['SimHei']
plt.rcParams["axes.unicode_minus"] = False
plt.ylim(3.8, 4.3)
x_width = range(0, len(x_data))
x2_width = [i + 0.2 for i in x_width]

# plt.bar(x_width, y_data, lw=0.5, color="#63b2ee", width=0.2, label="T-GCN")
plt.bar(x_width, y2_data, lw=0.5, color="#9987ce", width=0.2, label="GRU")
plt.bar(x2_width, y3_data, lw=0.5, color="#76da91", width=0.2, label="ET-GCN")

plt.xticks(range(0, 4), x_data)

# plt.title("RMSE")
plt.legend()
plt.xlabel("T/min")
plt.ylabel("RMSE")

# plt.show()

# plt.savefig(fname="rmse-1.eps", format="eps")
plt.savefig(fname="rmse-2.eps", format="eps")