import os
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# matplotlib.rcParams.update({'font.size': 15})
# plt.rc("font", family="Times New Roman")

# dir_path = 'D:/test/sub_iccv'
# dir_path = 'D:/test/sub_iccv_128'
# dir_path = 'D:/test/sub_iccv_512'
dir_path = 'D:/test/sub_iccv_512_origin'

log_filename = 'loss.csv'
dirs = os.listdir(dir_path)
int_pattern = re.compile('\d+')
float_pattern = re.compile('\d+\.\d+')

opt_list = []
train_loss_list = []
# test_loss_list = []
for filename in dirs:
    if os.path.isdir(os.path.join(dir_path, filename)):
        log_path = os.path.join(dir_path, filename, log_filename)
        train_loss = []
        test_loss = []
        with open(log_path, 'r') as f:
            for line in f.readlines()[1:]:
                # print(line)
                search_results = float_pattern.findall(line)
                # print(search_results)
                train_loss.append(float(search_results[1]))
                # test_loss.append(float(search_results[3]))
        train_loss_list.append(np.array(train_loss))
        opt_list.append(filename)
        # test_loss_list.append(np.array(test_loss))

x = np.arange(1, 1+len(train_loss_list[0]))

# draw train loss
plt.figure()
plt.grid(ls='--')
plt.xlabel('Epoch')
plt.ylabel('Train Loss')
# ax.set_xscale("log")
plt.yscale("log")
# plt.title("ResNet32 with SGD on Cifar10")
for i in range(len(opt_list)):
    plt.plot(range(len(train_loss_list[i])), train_loss_list[i], linewidth=2.05, label=opt_list[i])
    # plt.plot(range(6), train_loss_list[i][0:6], linewidth=2.05, label=opt_list[i])
plt.legend(loc='upper right')
# plt.show()
# plt.savefig(os.path.join(dir_path, 'train_loss.pdf'))
plt.savefig(os.path.join(dir_path, 'train_loss.jpg'))