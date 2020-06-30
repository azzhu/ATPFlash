#!/GPFS/zhangli_lab_permanent/zhuqingjie/env/py3/bin/python
'''
@Time    : 11.28 2019 下午 04:28
@Author  : qiujiwen
@User    : qiu
@FileName: find_connected_regions.py
@Software: PyCharm
'''
# coding: utf-8
import numpy as np
from collections import Counter
import cv2

import sys
import pickle

# import sys
# print(sys.maxsize)
# sys.setrecursionlimit(1000000000)
# sys.setrecursionlimit(10000000000)

# 递归深度最大值
# sys.setrecursionlimit(2147483647)
# sys.setrecursionlimit(1000000000)
# sys.setrecursionlimit(10000000)
# sys.setrecursionlimit(1000000000)
sys.setrecursionlimit(2147483646)
np.set_printoptions(threshold=np.inf)


class demo():
    def __init__(self, smallest_pixel, npy_file_path):
        # self.imgs = np.load('../data500.npy')
        if type(npy_file_path) is str:
            self.imgs = np.load(npy_file_path)
        else:
            self.imgs = npy_file_path
        print(self.imgs.shape)

        self.imgs = np.array(self.imgs, dtype=int)
        self.imgs_shape = self.imgs.shape
        print(self.imgs_shape)
        self.z = self.imgs_shape[0]
        self.x = self.imgs_shape[1]
        self.y = self.imgs_shape[2]
        self.smallest_pixel = smallest_pixel
        self.uniq1 = 1
        self.list1 = []
        self.list2 = []
        self.list3 = []

    def recursion(self, i, j, k):
        if self.imgs[i][j][k] == 255:
            self.list2.append([i, j, k])
            # print(self.list2)
            self.imgs[i][j][k] = self.uniq1
            for m in range(i - 1, i + 2):
                for n in range(j - 1, j + 2):
                    for q in range(k - 1, k + 2):
                        if ((not ((m == i) and (n == j) and (q == k))) and (
                                ((m != -1) and (n != -1) and (q != -1)) and (
                                (m != self.z) and (n != self.x) and (q != self.y)))):
                            self.recursion(m, n, q)
        return

    def preprocess(self):
        print(self.z)
        for i in range(self.z):
            # for i in range(200,self.z):
            print(i, end=',', flush=True)
            # if i==253:
            #     print()
            for j in range(self.x):
                for k in range(self.y):
                    if self.imgs[i][j][k] == 255:

                        self.list2 = []

                        if self.uniq1 == 255:
                            self.uniq1 = 0
                            self.list3 = self.list2
                            self.recursion(i, j, k)
                            self.uniq1 = 255
                        else:
                            self.recursion(i, j, k)

                        if len(self.list2) <= self.smallest_pixel:
                            for zero in self.list2:
                                self.imgs[zero[0]][zero[1]][zero[2]] = 0
                        else:
                            self.uniq1 = self.uniq1 + 1
                            self.list1.append(self.list2)

        if len(self.list3) > self.smallest_pixel:
            for zero in self.list2:
                self.imgs[zero[0]][zero[1]][zero[2]] = 255
        f = open('list.data', 'wb')
        pickle.dump(self.list1, f)
        f.close()
        # np.save('res1.npy', self.imgs)
        print('连通区域3D像素个数大于{}个的连通区域共有{}个'.format(self.smallest_pixel, len(self.list1)))
        return self.imgs


def denoise(smallest_pixel, img):
    g = demo(smallest_pixel, img)
    return g.preprocess()


if __name__ == '__main__':
    x = 'nihaoma'
    xb = b'nihaoma'
    print(x)
    print(xb)
    exit()
    # fist = True
    smallest_pixel = 150

    g = demo(smallest_pixel, '/home/zhangli_lab/zhuqingjie/dataset/atp_imaging/data2_kps_max.npy')
    g.preprocess()
    # g.denois()
    print('ok')
