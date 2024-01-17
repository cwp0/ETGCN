import random

import matplotlib.pyplot as plt
from pylab import xticks
import numpy as np

for i in range(0, 12):
    pre = []
    y = []
    with open("./predictions.txt", "r") as f:
        lines = f.readlines()[i*60:(i+1)*60]
        for line in lines:
            a, b = line[:-1].split(" ")
            a = int(a)
            b = int(b)
            # if b < 13 or (b >= 29 and b <= 31) or b >= 34:
            #     b = random.randint(20, 25)
            pre.append(a)
            y.append(b)
    test_pre = np.array(pre)
    test_y = np.array(y)

    # fig = plt.figure(figsize=(15, 8))
    # ax1 = fig.add_subplot(2, 1, 1)
    # ax2 = fig.add_subplot(2, 1, 2)
    xticks_label = [i * 3 for i in range(1, 61)]
    # xticks(np.linspace(0, 180, 6, endpoint=False), xticks_label)

    plt.plot(xticks_label, test_pre, "red", linestyle='-.', label="ET-GCN")
    plt.plot(xticks_label, test_y, "blue", linestyle='-', label="Truth")
    plt.xlabel("T/min")

    plt.ylabel("load")
    plt.legend(loc="upper right", fontsize=10)
    plt.savefig(fname="etgcn-load-" + str(i) +".eps", format="eps")
    plt.clf()

# pre = []
# y = []
# with open("./predictions.txt", "r") as f:
#     lines = f.readlines()[60:120]
#     for line in lines:
#         a, b = line[:-1].split(" ")
#         a = int(a)
#         b = int(b)
#         # if b < 13 or (b >= 29 and b <= 31) or b >= 34:
#         #     b = random.randint(20, 25)
#         pre.append(a)
#         y.append(b)
# test_pre = np.array(pre)
# test_y = np.array(y)
#
# # fig = plt.figure(figsize=(15, 8))
# # ax1 = fig.add_subplot(2, 1, 1)
# # ax2 = fig.add_subplot(2, 1, 2)
# xticks_label = [i * 3 for i in range(1, 61)]
# # xticks(np.linspace(0, 180, 6, endpoint=False), xticks_label)
#
# plt.plot(xticks_label, test_pre, "red", linestyle='-.', label="ET-GCN")
# plt.plot(xticks_label, test_y, "blue", linestyle='-', label="Truth")
# plt.xlabel("T/min")
#
# plt.ylabel("load")
# plt.legend(loc="upper right", fontsize=10)
# plt.show()
