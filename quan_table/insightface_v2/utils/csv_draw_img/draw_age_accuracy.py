import os
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# matplotlib.rcParams.update({'font.size': 15})
# plt.rc("font", family="Times New Roman")

# dir_path = 'D:/test/sub_iccv_128'
dir_path = 'D:/test/sub_iccv_512_origin'

log_filename = 'agedb_30_accuracy.csv'
dirs = os.listdir(dir_path)
int_pattern = re.compile('\d+')
float_pattern = re.compile('\d+\.\d+')

opt_list = []
lfw_accuracy_list = []
# test_loss_list = []
for filename in dirs:
    if os.path.isdir(os.path.join(dir_path, filename)):
        log_path = os.path.join(dir_path, filename, log_filename)
        lfw_accuracy = []
        test_loss = []
        with open(log_path, 'r') as f:
            for line in f.readlines()[1:]:
                # print(line)
                search_results = float_pattern.findall(line)
                # print(search_results)
                if 'auxnet' in filename:
                        lfw_accuracy.append(100 * float(search_results[1]))
                else:
                        lfw_accuracy.append(float(search_results[1]))
                # test_loss.append(float(search_results[3]))
        lfw_accuracy_list.append(np.array(lfw_accuracy))
        opt_list.append(filename)
        # test_loss_list.append(np.array(test_loss))

x = np.arange(1, 1+len(lfw_accuracy_list[0]))

# draw train loss
plt.figure()
plt.grid(ls='--')
plt.xlabel('Epoch')
plt.ylabel('age Accuracy')
# ax.set_xscale("log")
plt.yscale("log")
# plt.title("ResNet32 with SGD on Cifar10")
for i in range(len(opt_list)):
    plt.plot(range(len(lfw_accuracy_list[i])), lfw_accuracy_list[i], linewidth=2.05, label=opt_list[i])
plt.legend(loc='lower right')
# plt.show()
# plt.savefig(os.path.join(dir_path, 'age_accuracy.pdf')
plt.savefig(os.path.join(dir_path, 'age_accuracy.jpg'))