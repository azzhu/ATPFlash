#!/GPFS/zhangli_lab_permanent/zhuqingjie/env/py3/bin/python
'''
@Time    : 10.21 0021 下午 02:38
@Author  : zhuqingjie 
@User    : zhu
@FileName: main.py
@Software: PyCharm
'''
import cv2, random, os
import tensorflow as tf
import tensorflow.keras.layers as KL
import numpy as np
from PIL import Image
# from multiprocessing import Pool
import time
from skimage import io
import matplotlib.pyplot as plt
import pickle
from find_connected_regions import denoise
from pathlib import Path


# 处理一张3d图像，返回2d图像list，16为图像改成了8位
def read3dtif(p):
    tiff = Image.open(p, mode='r')
    print(tiff.size)
    print(tiff.n_frames)
    img_list = []
    for i in range(tiff.n_frames):
        tiff.seek(i)
        img = np.array(tiff)
        if img.dtype == np.uint16:
            img = img.astype(np.float)
            img = img / 65535. * 255
            img = img.astype(np.uint8)
        img_list.append(img)
    return img_list


class OD(object):
    '''
    这一版的问题：这一版只是标记出了比较亮的地方，但是跑偏了，目标要的是z轴方向某种规律突变的地方。
    '''

    def __init__(self, src_img_path, kernel_size, dst_img_path, is_dir):
        self.src_img_path = src_img_path
        self.kernel_size = kernel_size  # only odd number
        self.dst_img_path = dst_img_path
        self.th = 70
        self.is_dir = is_dir

    def load_img(self):
        print('load_img...')
        if not self.is_dir:
            npy_path = self.src_img_path[:-3] + 'npy'
            if not os.path.exists(npy_path):
                self.img = np.array(read3dtif(self.src_img_path))
                np.save(npy_path, self.img)
            self.img = np.load(npy_path)
            # print(self.img.shape)
            # self.img = self.img[0:500]
            self.img = self.img
            print(self.img.shape)
            self.img = np.transpose(self.img, (1, 2, 0))
            print(self.img.shape)
        else:
            dir = Path(self.src_img_path)
            lens = len(list([fn for fn in dir.iterdir() if '.tif' in str(fn)]))
            fns = [f'{self.src_img_path}/1 ({i}).tif' for i in range(1, lens + 1)]
            fns = [f for f in fns if os.path.exists(f)]
            # fns = sorted(fns)
            # imgspath = [Path(dir.parent, fn) for fn in fns]
            self.img = np.array([cv2.imread(str(f), 0) for f in fns])
            print(self.img.shape)
            self.img = np.transpose(self.img, (1, 2, 0))
        print('load_img... ok ')

    def smooth(self):
        def smooth_op(x):
            k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            k2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (7, 7))
            close_img = cv2.morphologyEx(x, cv2.MORPH_CLOSE, k1)
            open_img = cv2.morphologyEx(close_img, cv2.MORPH_OPEN, k2)
            open_img = cv2.morphologyEx(open_img, cv2.MORPH_OPEN, k1)
            return open_img

        dst_list = []
        for img in self.dst:
            dst_list.append(smooth_op(img))
        self.dst = np.array(dst_list)

    def object_detection(self):
        x = tf.constant(self.img[np.newaxis, :, :, :, np.newaxis], dtype=tf.float16)
        # x = tf.layers.conv3d(x, 1, self.kernel_size, use_bias=False, kernel_initializer=tf.initializers.ones,
        #                      trainable=False, padding='same')
        print(tf.reduce_max(x))
        print(tf.reduce_min(x))
        ker = np.ones((self.kernel_size, self.kernel_size, self.kernel_size), dtype=np.float16) / (
                self.kernel_size ** 3)
        t1 = time.time()
        x = KL.Conv3D(1, self.kernel_size, use_bias=False, kernel_initializer=tf.initializers.constant(value=ker),
                      trainable=False, padding='same', activation=tf.nn.relu)(x)
        print(f'conv3d timeuse:{time.time() - t1}')
        print(tf.reduce_max(x))
        print(tf.reduce_min(x))
        dst = x.numpy()
        dst = dst[0, :, :, :, 0]
        dst = np.transpose(dst, (2, 0, 1))
        dst = dst >= self.th
        dst = dst.astype(np.uint8)
        self.dst = dst * 255
        t1 = time.time()
        self.smooth()
        print(f'smooth timeuse:{time.time() - t1}')
        print(self.dst.max())
        np.save('temp/dst.npy', self.dst)
        io.imsave(self.dst_img_path, self.dst)

    def renderer(self):
        r = np.array(self.img, dtype=np.float32)
        r = np.transpose(r, (2, 0, 1))
        mask = np.load('temp/dst.npy')
        mask = mask.astype(np.float32)
        r2 = mask
        mask = mask / 255
        # and_ = r * mask
        neg_ = mask == 0
        neg_ = neg_.astype(np.float32)
        gb = r * neg_
        rgb = np.array((r2, gb, gb), np.uint8)
        dst = np.transpose(rgb, [1, 2, 3, 0])
        io.imsave('/home/zhangli_lab/zhuqingjie/dataset/temp/Muti_dst_rgb.tif', dst)


class OD2(OD):
    '''
    更正第一版的问题
    '''

    def __init__(self, src_img_dir, der_param=9, der_th=21, smooth_ker=7):
        super().__init__(src_img_dir, 3, None, False)
        # self.h = 19  # 求导的跨度大小，(f(x+h)-f(x))/h
        # self.smooth_ker = 5
        # self.der_th = 45
        self.h = der_param  # 求导的跨度大小，(f(x+h)-f(x))/h
        self.smooth_ker = smooth_ker
        self.der_th = der_th

    # def test(self):
    #     # # part 1
    #     # self.load_img()
    #     # x = tf.constant(self.img[np.newaxis, :, :, :, np.newaxis], dtype=tf.float16)
    #     # skh = int(self.smooth_ker / 2)
    #     # ker_l = self.h + 2 * skh
    #     # ker = np.zeros((1, 1, ker_l), dtype=np.float16) / ker_l
    #     # ker[:, :, :self.smooth_ker] = -1 / self.smooth_ker
    #     # ker[:, :, -self.smooth_ker:] = 1 / self.smooth_ker
    #     # t1 = time.time()
    #     # x = KL.Conv3D(1, (1, 1, ker_l), use_bias=False,
    #     #               kernel_initializer=tf.initializers.constant(value=ker),
    #     #               trainable=False, padding='valid')(x)
    #     # print(f'conv3d timeuse:{time.time() - t1}')
    #     # print(tf.reduce_max(x))
    #     # print(tf.reduce_min(x))
    #     # dst = x.numpy()
    #     # dst = dst[0, :, :, :, 0]
    #     # print(dst.shape)
    #     # np.save('temp/der.npy', dst)
    #     # dst = np.transpose(dst, (2, 0, 1))
    #     # # dst = np.transpose(dst, (1, 2, 0))
    #     # # dst = dst[:, 155, :]
    #     # dst = 255. / (dst.max() - dst.min()) * (dst - dst.min())
    #     # # dst = dst[20:]
    #     # dst = dst.astype(np.uint8)
    #     # io.imsave('temp/der.tif', dst)
    #     # # cv2.imwrite('temp/der.tif', dst)
    #
    #     # # part2
    #     # x = np.load('temp/der.npy')
    #     # x = np.transpose(x, (2, 0, 1))
    #     # max_ps = x > 45
    #     # max_ps = max_ps.astype(np.uint8) * 255
    #     # io.imsave('temp/max_ps.tif', max_ps)
    #     # np.save('temp/max_ps.npy', max_ps)
    #     # max_ps = max_ps.astype(np.int32)
    #     # max_ps = np.sum(max_ps, 0)
    #     # max_ps = max_ps.astype(np.bool).astype(np.uint8) * 255
    #     # cv2.imwrite('temp/max_ps.bmp', max_ps)
    #     # print()
    #
    #     max_ps = np.load('temp/max_ps.npy')
    #     min_ps = np.load('temp/min_ps.npy')
    #     # max_ps = max_ps[:, :, :, np.newaxis]
    #     # min_ps = min_ps[:, :, :, np.newaxis]
    #     zeros = np.zeros_like(max_ps)
    #     res = np.stack((max_ps, min_ps, min_ps), -1)
    #     io.imsave('temp/res_rgb.tif', res)

    def calc_derivative(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        # tf.enable_eager_execution()
        self.load_img()
        x = tf.constant(self.img[np.newaxis, :, :, :, np.newaxis], dtype=tf.float16)
        skh = int(self.smooth_ker / 2)
        ker_l = self.h + 2 * skh
        ker = np.zeros((1, 1, ker_l), dtype=np.float16) / ker_l
        ker[:, :, :self.smooth_ker] = -1 / self.smooth_ker
        ker[:, :, -self.smooth_ker:] = 1 / self.smooth_ker
        t1 = time.time()
        x = KL.Conv3D(1, (1, 1, ker_l), use_bias=False,
                      kernel_initializer=tf.initializers.constant(value=ker),
                      trainable=False, padding='valid')(x)
        print(f'conv3d timeuse:{time.time() - t1}')
        print(tf.reduce_max(x))
        print(tf.reduce_min(x))
        dst = x.numpy()
        dst = dst[0, :, :, :, 0]
        print(dst.shape)
        self.der_img = dst
        # der_path = self.src_img_path[:-4] + '_der.npy'
        # np.save(der_path, dst)
        # dst = np.transpose(dst, (2, 0, 1))
        # dst = 255. / (dst.max() - dst.min()) * (dst - dst.min())
        # dst = dst.astype(np.uint8)
        # der_img_path = self.src_img_path[:-4] + '_der_img.tif'
        # io.imsave(der_img_path, dst)

    def view_distribution(self):
        der_path = self.src_img_path[:-4] + '_der.npy'
        der = np.load(der_path)
        der = der.flatten()
        plt.hist(der, bins=200)
        plt.savefig('temp/hist.jpg')

    def calc_keypoints(self):
        # der_path = self.src_img_path[:-4] + '_der.npy'
        # der = np.load(der_path)
        der = self.der_img
        der = np.transpose(der, (2, 0, 1))
        max_ps = der > self.der_th
        max_ps = max_ps.astype(np.uint8) * 255
        # max_ps_imgpath = os.path.join(os.path.dirname(der_path), 'data2_kps_max.npy')
        # # io.imsave(max_ps_imgpath, max_ps)
        # np.save(max_ps_imgpath, max_ps)
        max_ps_denoise = denoise(150, max_ps)
        self.max_ps_denoise = max_ps_denoise
        # exit()
        #
        # min_ps = der < -25
        # min_ps = min_ps.astype(np.uint8) * 255
        # # xy轴投射
        # x = max_ps.astype(np.int32)
        # x = np.sum(x, 0)
        # x = x.astype(np.bool).astype(np.uint8) * 255
        # cv2.imwrite('temp/max_ps.bmp', x)
        # x = min_ps.astype(np.int32)
        # x = np.sum(x, 0)
        # x = x.astype(np.bool).astype(np.uint8) * 255
        # cv2.imwrite('temp/min_ps.bmp', x)
        # # 彩色渲染
        # res = np.stack((max_ps, min_ps, min_ps), -1)
        # kps_path = self.src_img_path[:-4] + '_kps_rgb.tif'
        # io.imsave(kps_path, res)
        # kps_npy_path = self.src_img_path[:-4] + '_kps_rgb.npy'
        # np.save(kps_npy_path, res)

    def img_merge(self):
        # 在y轴方向合并，目的是为了方便对比观察
        self.load_img()
        img1 = np.transpose(self.img, (2, 0, 1))
        img1 = np.stack((img1, img1, img1), -1)
        kps_npy_path = self.src_img_path[:-4] + '_kps_rgb.npy'
        img2 = np.load(kps_npy_path)
        pad = int((img1.shape[0] - img2.shape[0]) / 2)
        img2 = np.pad(img2, ((pad, pad), (0, 0), (0, 0), (0, 0)), 'constant',
                      constant_values=((0, 0), (0, 0), (0, 0), (0, 0)))
        img_new = np.concatenate((img2, img1), -2)
        merge_img_path = self.src_img_path[:-4] + '_merge_img.tif'
        io.imsave(merge_img_path, img_new)
        pass

    # 求出连通区域之后，转成txt格式
    def CRS2txt(self):
        # crs = np.load(os.path.join(os.path.dirname(self.src_img_path), 'data2_kps_max_CRS_denois.npy'))
        crs = self.max_ps_denoise
        ma = crs.max()
        zz, hh, ww = crs.shape

        # 遍历求连通区域像素点集合
        dcrs = dict()
        for i in range(1, ma + 1): dcrs[i] = []
        print(zz)
        for z in range(zz):
            print(f'{z}', end=',', flush=True)
            for h in range(hh):
                for w in range(ww):
                    if crs[z, h, w] != 0:
                        dcrs[crs[z, h, w]].append((z, h, w))
        pickle.dump(dcrs, open('dcrs_dict.pkl', 'wb'))

        # dcrs = pickle.load(open('dcrs_dict.pkl', 'rb'))
        nb = 0  #
        f = open('res.txt', 'w')
        for i in range(1, len(dcrs) + 1):
            # print(i, flush=True)
            d = dcrs[i]
            if len(d) == 0: continue
            xyps = set()
            xyps.clear()
            zps = []  # 要算z轴重心，所以不用集合用list
            for xyz in d:
                xyps.add(str(xyz[1:]))
                zps.append(xyz[0])
            xyps_str = ','.join(xyps)
            xyps_nb = len(xyps)
            zps_cp = int(np.mean(zps))
            f.write(f'{zps_cp};{xyps_nb};{xyps_str}\n')
            nb += 1
        print(f'nb:{nb}')
        f.close()

    def CRS2txt_test(self):
        lines = open('res.txt', 'r').readlines()
        # img = np.load('/home/zhangli_lab/zhuqingjie/dataset/atp_imaging/data2.npy')
        img = np.transpose(self.img, [2, 0, 1])
        print(img.shape)
        img_zeros = np.zeros_like(img)
        img = np.stack([img, img_zeros, img], -1)
        for k, line in enumerate(lines):
            print(f'{k}/{len(lines)}', end=',', flush=True)
            z = int(line.split(';')[0])
            ps = line.split(';')[-1].split('),(')[1:-1]
            ps = [(int(p.split(',')[0]), int(p.split(',')[1])) for p in ps]
            for p in ps:
                img[z, p[0], p[1]] = [0, 255, 0]
        io.imsave('/home/zhangli_lab/zhuqingjie/dataset/atp_imaging/data2_txt_res_show.tif', img)

    def run(self):
        print('\ncalc_derivative...')
        self.calc_derivative()
        print('\ncalc_keypoints...')
        self.calc_keypoints()
        print('\nCRS2txt...')
        self.CRS2txt()
        print('\nCRS2txt_test...')
        # self.CRS2txt_test()
        print('done.')


if __name__ == '__main__':
    t1 = time.time()
    od = OD2(
        src_img_dir='/home/zhangli_lab/zhuqingjie/DATA/chenyue/lps12.tif.frames',
        der_param=9,
        der_th=21,
        smooth_ker=7,
    )
    od.run()
    print(f'time_used:{(time.time() - t1) / 60}')
