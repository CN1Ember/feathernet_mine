#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/15 10:29
# @Author  : xiezheng
# @Site    : 
# @File    : ice_draw_rocs.py

import numpy as np
import matplotlib.pyplot as plt

# show roc curve
def main():
    file_name_list = [ "SphereMobileNet_v2-origin-012201.npz",]
                      # "SphereNet-pruned_65_012001.npz"]
    line_color = ["red"]# , "green"]
    line_label = ["origin"]# , "prune 65%"]

    for i, file_name in enumerate(file_name_list):
        data = np.load(file_name)
        # print(data)
        # print(data['arr_0'])
        # print(data['arr_1'])
        far = data['arr_0']
        veri_far = data['arr_1']
        plt.semilogx(far, veri_far, color=line_color[i],
                     linewidth=2.0,
                     label=line_label[i])
    plt.grid(True)
    plt.legend(loc=0)
    plt.xlabel('False Accept Rate')
    plt.ylabel('Verification Rate')
    plt.title('Face Verification ROC Curve')
    plt.show()
    # plt.savefig('Sphereface4_vggface_ROC.png')


if __name__ == '__main__':
    main()