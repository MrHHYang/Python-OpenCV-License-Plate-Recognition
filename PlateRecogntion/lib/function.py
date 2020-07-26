#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import lib.math as img_math
import lib.recognition as img_recognition

SZ = 20  # 训练图片长宽
MAX_WIDTH = 1000  # 原始图片最大宽度
Min_Area = 2000  # 车牌区域允许最大面积
PROVINCE_START = 1000


def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class StatModel(object):
    def load(self, fn):
        self.model = self.model.load(fn)

    def save(self, fn):
        self.model.save(fn)


class SVM(StatModel):
    def __init__(self, C=1, gamma=0.5):
        self.model = cv2.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)

    # 训练svm
    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    # 字符识别
    def predict(self, samples):
        r = self.model.predict(samples)
        return r[1].ravel()


class CardPredictor:
    def __init__(self):
        pass

    def train_svm(self):
        '''生成dat文件，完成机器学习算子，此步操作在main文件执行时，就被自动调用了'''
        # 识别英文字母和数字
        self.model = SVM(C=1, gamma=0.5)
        # 识别中文
        self.modelchinese = SVM(C=1, gamma=0.5)
        if os.path.exists("lib/svm.dat"):
            self.model.load("lib/svm.dat")
        if os.path.exists("lib/svmchinese.dat"):
            self.modelchinese.load("lib/svmchinese.dat")

    def img_first_pre(self, car_pic_file):
        """
        返回Otsu’s二值化边缘化的新图像和原图，需要两个变量接收
        :param car_pic_file: 图像文件
        :return:已经处理好的图像文件 原图像文件
        """
        if type(car_pic_file) == type(""):
            img = img_math.img_read(car_pic_file)
        else:
            img = car_pic_file

        pic_hight, pic_width = img.shape[:2]
        if pic_width > MAX_WIDTH:
            resize_rate = MAX_WIDTH / pic_width
            img = cv2.resize(img, (MAX_WIDTH, int(pic_hight * resize_rate)), interpolation=cv2.INTER_AREA)
        # cv_show("img", img)
        # 缩小图片

        blur = 5
        img = cv2.GaussianBlur(img, (blur, blur), 0)
        # cv_show("blur=5", img)
        oldimg = img
        return img, oldimg

    def color_and_contour(self, filename, oldimg, img_contours):
        """
        color_and_contour()方法是通过颜色和形状定位车牌位置的方法
        返回识别到的字符、定位的车牌图像、车牌颜色
        :param filename: 图像文件
        :param oldimg: 原图像文件
        :return: 返回识别到的字符、定位的车牌图像、车牌颜色
        """
        pic_hight, pic_width = img_contours.shape[:2]
        lower_blue = np.array([100, 110, 110])
        # 在 OpenCV 的 HSV 格式中，H（色彩/色度）的取值范围是 [0，179]，S（饱和度）的取值范围 [0，255]，V（亮度）的取值范围 [0，255]。
        upper_blue = np.array([130, 255, 255])
        lower_yellow = np.array([15, 55, 55])
        upper_yellow = np.array([50, 255, 255])
        lower_green = np.array([50, 50, 50])
        upper_green = np.array([100, 255, 255])
        hsv = cv2.cvtColor(filename, cv2.COLOR_BGR2HSV)
        # cv_show('hsv', hsv)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        # 第一个参数：hsv指的是原图
        # 第二个参数：lower_red指的是图像中低于这个lower_red的值，图像值变为0
        # 第三个参数：upper_red指的是图像中高于这个upper_red的值，图像值变为0
        # 而在lower_red～upper_red之间的值变成255
        # cv_show('mask_blue', mask_blue)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        # cv_show('mask_yellow', mask_yellow)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        # cv_show('mask_green', mask_green)
        output = cv2.bitwise_and(hsv, hsv, mask=mask_blue + mask_yellow + mask_green)
        # https://www.cnblogs.com/Undo-self-blog/p/8434906.html
        # 将hsv与mask的像素值想与
        # cv_show('output', output)
        # 根据阈值找到对应颜色

        output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        # cv_show('output', output)
        Matrix = np.ones((20, 20), np.uint8)
        img_edge1 = cv2.morphologyEx(output, cv2.MORPH_CLOSE, Matrix)
        img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, Matrix)
        # cv_show('img_edge2',img_edge2)

        card_contours = img_math.img_findContours(img_edge2)
        card_imgs = img_math.img_Transform(card_contours, oldimg, pic_width, pic_hight)
        # print(card_imgs.shape)
        colors, card_imgs = img_math.img_color(card_imgs)
        # print(card_imgs)

        predict_result = []
        predict_str = ""
        roi = None
        card_color = None

        for i, color in enumerate(colors):
            # print('i的值：', i) i = 0

            if color in ("blue", "yello", "green"):
                card_img = card_imgs[i]
                # cv_show('card_img', card_img)

                try:
                    gray_img = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
                    # cv_show('gray_img0',gray_img)
                except:
                    print("gray转换失败")

                # 黄、绿车牌字符比背景暗、与蓝车牌刚好相反，所以黄、绿车牌需要反向
                if color == "green" or color == "yello":
                    gray_img = cv2.bitwise_not(gray_img)
                ret, gray_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                print('二值化阈值为：', ret)
                # cv_show('gray_img1',gray_img)
                x_histogram = np.sum(gray_img, axis=1)
                print('x_histogram的值为：--', x_histogram)

                x_min = np.min(x_histogram)
                x_average = np.sum(x_histogram) / x_histogram.shape[0]
                x_threshold = (x_min + x_average) / 2
                wave_peaks = img_math.find_waves(x_threshold, x_histogram)
                if len(wave_peaks) == 0:
                    # print("peak less 0:")
                    continue
                # 认为水平方向，最大的波峰为车牌区域
                wave = max(wave_peaks, key=lambda x: x[1] - x[0])
                gray_img = gray_img[wave[0]:wave[1]]
                # cv_show('gray_img2', gray_img)
                # 查找垂直直方图波峰
                row_num, col_num = gray_img.shape[:2]
                # 去掉车牌上下边缘1个像素，避免白边影响阈值判断
                gray_img = gray_img[1:row_num - 1]
                # cv_show('gray_img3', gray_img)
                y_histogram = np.sum(gray_img, axis=0)
                y_min = np.min(y_histogram)
                y_average = np.sum(y_histogram) / y_histogram.shape[0]
                y_threshold = (y_min + y_average) / 5  # U和0要求阈值偏小，否则U和0会被分成两半
                wave_peaks = img_math.find_waves(y_threshold, y_histogram)
                if len(wave_peaks) < 6:
                    # print("peak less 1:", len(wave_peaks))
                    continue

                wave = max(wave_peaks, key=lambda x: x[1] - x[0])
                max_wave_dis = wave[1] - wave[0]
                # 判断是否是左侧车牌边缘
                if wave_peaks[0][1] - wave_peaks[0][0] < max_wave_dis / 3 and wave_peaks[0][0] == 0:
                    wave_peaks.pop(0)

                # 组合分离汉字
                cur_dis = 0
                for i, wave in enumerate(wave_peaks):
                    if wave[1] - wave[0] + cur_dis > max_wave_dis * 0.6:
                        break
                    else:
                        cur_dis += wave[1] - wave[0]
                if i > 0:
                    wave = (wave_peaks[0][0], wave_peaks[i][1])
                    wave_peaks = wave_peaks[i + 1:]
                    wave_peaks.insert(0, wave)

                point = wave_peaks[2]
                point_img = gray_img[:, point[0]:point[1]]
                # cv_show('point_img', point_img)
                if np.mean(point_img) < 255 / 5:
                    wave_peaks.pop(2)

                if len(wave_peaks) <= 6:
                    # print("peak less 2:", len(wave_peaks))
                    continue

                part_cards = img_math.seperate_card(gray_img, wave_peaks)

                for i, part_card in enumerate(part_cards):
                    # 可能是固定车牌的铆钉

                    if np.mean(part_card) < 255 / 5:
                        # print("a point")
                        continue
                    part_card_old = part_card

                    w = abs(part_card.shape[1] - SZ) // 2

                    part_card = cv2.copyMakeBorder(part_card, 0, 0, w, w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                    # cv_show('part_card', part_card)
                    part_card = cv2.resize(part_card, (SZ, SZ), interpolation=cv2.INTER_AREA)
                    # cv_show('part_card2', part_card)
                    part_card = img_recognition.preprocess_hog([part_card])
                    # cv_show('part_card3', part_card)
                    if i == 0:
                        resp = self.modelchinese.predict(part_card)
                        charactor = img_recognition.provinces[int(resp[0]) - PROVINCE_START]
                    else:
                        resp = self.model.predict(part_card)
                        charactor = chr(resp[0])
                    # 判断最后一个数是否是车牌边缘，假设车牌边缘被认为是1
                    if charactor == "1" and i == len(part_cards) - 1:
                        if part_card_old.shape[0] / part_card_old.shape[1] >= 7:  # 1太细，认为是边缘
                            continue
                    predict_result.append(charactor)
                    predict_str = "".join(predict_result)

                roi = card_img
                card_color = color
                break
        return predict_str, roi, card_color  # 识别到的字符、定位的车牌图像、车牌颜色
