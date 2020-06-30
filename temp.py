#!/GPFS/zhangli_lab_permanent/zhuqingjie/env/py3/bin/python
'''
@Time    : 10.23 0023 下午 03:56
@Author  : zhuqingjie 
@User    : zhu
@FileName: temp.py
@Software: PyCharm
'''
# import tensorflow as tf
# import tensorflow.keras.layers as KL
import numpy as np
from skimage import io

src_img_path = '/home/zhangli_lab/zhuqingjie/dataset/atp_imaging/data2_der.npy'
x = np.load(src_img_path)
x = x[:, :, 1:501]
x = np.transpose(x, (2, 0, 1))
x = x > 45
x = x.astype(np.uint8) * 255
print(x.max())
# io.imsave('/home/zhangli_lab/zhuqingjie/dataset/atp_imaging/data500.tif', x)
np.save('/home/zhangli_lab/zhuqingjie/dataset/atp_imaging/data500.npy', x)
print()
