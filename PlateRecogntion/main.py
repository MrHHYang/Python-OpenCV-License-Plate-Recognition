#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading
import time
import tkinter as tk
import cv2
import lib.function as predict
import lib.math as img_math
import lib.sql as img_sql
from lib.api import api_pic
from threading import Thread
from tkinter import ttk
from tkinter.filedialog import *
from PIL import Image, ImageTk, ImageGrab
import tkinter.messagebox

from hyperlpr import *

def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

class HyperLPR_PlateRecogntion(object):
    pass


class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, *, daemon=None):
        Thread.__init__(self, group, target, name, args, kwargs, daemon=daemon)
        self._return1 = None
        self._return2 = None
        self._return3 = None

    def run(self):
        if self._target is not None:
            try:
                self._return1, self._return2, self._return3 = self._target(*self._args, **self._kwargs)
            except:
                pass

    def join(self):
        Thread.join(self)
        return self._return1, self._return2, self._return3


class Surface(ttk.Frame):
    pic_path = ""
    viewhigh = 600
    viewwide = 600
    update_time = 0
    thread = None
    thread_run = False
    camera = None
    pic_source = ""
    color_transform = {"green": ("绿牌", "#55FF55"), "yello": ("黄牌", "#FFFF00"), "blue": ("蓝牌", "#6666FF")}
    # 用于后续的将返回的"blue"的key对应成value打印出来

    def __init__(self, win):
        """初始化函数，用于初始化方法、属性、excel表、数据库等"""
        ttk.Frame.__init__(self, win)
        frame_left = ttk.Frame(self)  # 创建四个容器frame_left、frame_right1、frame_right2、top
        frame_right1 = ttk.Frame(self)
        frame_right2 = ttk.Frame(self)
        win.title("车牌识别监测报警系统")
        win.minsize(850, 700)
        self.center_window()
        self.pic_path3 = ""
        self.cameraflag = 0

        self.pack(fill=tk.BOTH, expand=tk.YES, padx="10", pady="10")  # 放置对象
        frame_left.pack(side=LEFT, expand=1)
        frame_right1.pack(side=TOP, expand=1, fill=tk.Y)
        frame_right2.pack(side=RIGHT, expand=0)

        self.image_ctl = ttk.Label(frame_left)   # 创建一个标签image_ctl贴在容器frame_left上
        self.image_ctl.pack(anchor="nw")  # 锚定位（anchor ），加 padx, pady , 可将组件安排在指定位置

        # 右上角的容器部署
        ttk.Label(frame_right1, text='定位车牌位置：').grid(column=0, row=0, sticky=tk.W)
        self.roi_ct2 = ttk.Label(frame_right1)
        self.roi_ct2.grid(column=0, row=1, sticky=tk.W)

        ttk.Label(frame_right1, text='定位识别结果：').grid(column=0, row=2, sticky=tk.W)
        self.r_ct2 = ttk.Label(frame_right1, text="", font=('Times', '20'))
        self.r_ct2.grid(column=0, row=3, sticky=tk.W)
        self.color_ct2 = ttk.Label(frame_right1, text="", width="20")
        self.color_ct2.grid(column=0, row=4, sticky=tk.W)

        ttk.Label(frame_right1, text='-------------------------------').grid(column=0, row=5, sticky=tk.W)

        from_pic_ctl = ttk.Button(frame_right2, text="选择图片", width=20, command=self.from_pic)
        from_pic_ctl2 = ttk.Button(frame_right2, text="路径批量识别", width=20, command=self.from_pic2)
        from_vedio_ctl = ttk.Button(frame_right2, text="打开/关闭摄像头", width=20, command=self.from_vedio)
        from_video_ctl = ttk.Button(frame_right2, text="拍照并识别", width=20, command=self.video_pic)
        clean_ctrl = ttk.Button(frame_right2, text="清除识别数据", width=20, command=self.clean)
        camera_ctrl = ttk.Button(frame_right2, text="开关摄像头实时识别", width=20, command=self.camera_flag)

        # 放置按钮
        camera_ctrl.pack(anchor="se", pady="5")  # 开关摄像头实时识别
        from_vedio_ctl.pack(anchor="se", pady="5")  # 打开/关闭摄像头
        from_video_ctl.pack(anchor="se", pady="5")  # 拍照并识别
        from_pic_ctl2.pack(anchor="se", pady="5")  # 路径批量识别
        from_pic_ctl.pack(anchor="se", pady="5")  # 来自图片
        clean_ctrl.pack(anchor="se", pady="5")  # 清除识别数据
        # 右上角的容器部署结束

        self.clean()
        self.apistr = None
        img_sql.create_sql()  # 调用create_sql()方法，创建数据库表

        self.predictor = predict.CardPredictor()  # 调用lib.function下的类CardPredictor()创建对象predictor
        self.predictor.train_svm()  # 调用lib.function下的train_svm()方法

    def reset(self):
        """调用clean()方法，清除信息，重置属性"""
        win.geometry("850x700")
        self.clean()
        self.thread_run7 = False
        self.count = 0
        self.center_window()

    def center_window(self):
        """窗口中心化"""
        screenwidth = win.winfo_screenwidth()
        screenheight = win.winfo_screenheight()
        win.update()
        width = win.winfo_width()
        height = win.winfo_height()
        size = '+%d+%d' % ((screenwidth - width)/2, (screenheight - height)/2)
        win.geometry(size)

    def get_imgtk(self, img_bgr):
        """返回一个与tkinter兼容的照片图像imgtk"""
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(img)
        # Image.fromarray()从导出数组接口的对象（使用缓冲区协议）创建图像内存。
        w, h = im.size
        pil_image_resized = self.resize2(w, h, im)  # 调用resize2()方法进行图像大小格式化
        imgtk = ImageTk.PhotoImage(image=pil_image_resized)
        # 返回一个与tkinter兼容的照片图像imgtk。这可以在任何Tkinter需要图像对象的地方使用
        return imgtk

    def resize(self, w, h, pil_image):
        """将经过处理的图像的大小格式化"""
        w_box = 200
        h_box = 50
        f1 = 1.0*w_box/w
        f2 = 1.0*h_box/h
        factor = min([f1, f2])
        width = int(w*factor)
        height = int(h*factor)
        return pil_image.resize((width, height), Image.ANTIALIAS)

    def resize2(self, w, h, pil_image):
        """将经过处理的图像的大小格式化"""
        width = win.winfo_width()
        height = win.winfo_height()
        w_box = width - 250
        h_box = height - 100
        f1 = 1.0*w_box/w
        f2 = 1.0*h_box/h
        factor = min([f1, f2])
        width = int(w*factor)
        height = int(h*factor)
        return pil_image.resize((width, height), Image.ANTIALIAS)

    def show_roi2(self, r, roi, color):
        """将定位到的车牌图像展示在右边中间--颜色定位车牌位置--颜色定位识别结果处，其他注释参照show_roi1"""
        if r:
            try:
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                roi = Image.fromarray(roi)
                w, h = roi.size
                pil_image_resized = self.resize(w, h, roi)
                self.tkImage2 = ImageTk.PhotoImage(image=pil_image_resized)
                self.roi_ct2.configure(image=self.tkImage2, state='enable')
            except:
                pass
            self.r_ct2.configure(text=str(r))
            self.update_time = time.time()
            try:
                c = self.color_transform[color]
                self.color_ct2.configure(text=c[0], state='enable')
            except:
                self.color_ct2.configure(state='disabled')
        elif self.update_time + 8 < time.time():

            self.roi_ct2.configure(state='disabled')
            self.r_ct2.configure(text="")
            self.color_ct2.configure(state='disabled')

    def camera_flag(self):
        """打开摄像头，对拍摄到的每一帧进行实时识别"""
        if not self.thread_run:
            tkinter.messagebox.showinfo('提示', '请点击[打开摄像头]按钮！')
            return
        if not self.cameraflag:
            self.cameraflag = 1
            self.thread2 = threading.Thread(target=self.video_pic2)
            # 启动线程thread2，通过run()方法，调用video_pic2
            self.thread2.setDaemon(True)
            self.thread2.start()
            self.thread_run2 = True
        else:
            self.cameraflag = 0
            self.thread_run2 = False
            print("关闭摄像头实时识别 self.cameraflag", self.cameraflag)

    def from_vedio(self):
        """判断摄像头的状态，并打开摄像头，新建线程thread并调用方法vedio_thread()"""
        # camera参数初始为None
        if self.thread_run:
            if self.camera.isOpened():
                self.camera.release()
                print("关闭摄像头")
                self.camera = None
                self.thread_run = False
            return
        if self.camera is None:
            self.camera = cv2.VideoCapture(1)
            if not self.camera.isOpened():
                self.camera = None
                print("没有外置摄像头")
                self.camera = cv2.VideoCapture(0)
                if not self.camera.isOpened():
                    print("没有内置摄像头")
                    tkinter.messagebox.showinfo('警告', '摄像头打开失败！')
                    self.camera = None
                    return
                else:
                    print("打开内置摄像头")
            else:
                print("打开外置摄像头")
        self.pic_source = "摄像头"
        self.cameraflag = 0
        self.thread = threading.Thread(target=self.vedio_thread)
        self.thread.setDaemon(True)
        self.thread.start()
        self.thread_run = True

    def pic(self, pic_path):
        """对图像进行处理，处理完成后在主窗口左边显示要识别的图像，
        在右边展示--形状定位车牌位置--形状定位识别结果--颜色定位车牌位置--颜色定位识别结果--的信息"""
        self.apistr = None  # self.apistr字段表示的是，调用api进行车牌识别返回的车牌完整信息的字符串
        img_bgr = img_math.img_read(pic_path)  # 以uint8方式读取pic_path返回给img_bgr，即返回以uint8方式读取的彩色照片
        first_img, oldimg = self.predictor.img_first_pre(img_bgr)
        # 返回Otsu’s二值化边缘化的新图像和原图，需要两个变量接收
        # 其中first_img接收Otsu’s二值化边缘化的新图像
        if not self.cameraflag:
            # 这个if的作用相当于vedio_thread()，即传入的图像不是由相机拍摄的图像，将图像显示在主窗口左边
            # self.cameraflag置为0时，即传入的图像不是由相机拍摄的图像，将图像显示在主窗口左边
            self.imgtk = self.get_imgtk(img_bgr)
            self.image_ctl.configure(image=self.imgtk)  # 将图像显示在主窗口左边
        th2 = ThreadWithReturnValue(target=self.predictor.color_and_contour, args=(oldimg, oldimg, first_img))
        # 创建线程th2，通过run()方法回调color_and_contour()方法，并将args作为参数传入color_and_contour()方法中
        th2.start()
        # 开启线程th2，将color_and_contour()方法返回的结果存储到变量r_color, roi_color, color_color中
        r_color, roi_color, color_color = th2.join()

        try:
            Plate = HyperLPR_PlateRecogntion(img_bgr)
            # HyperLPR_PlateRecogntion是未完成的通过HyperLPR库进行车牌识别的更高版本
            # print(Plate[0][0])
            r_c = Plate[0][0]
            r_color = Plate[0][0]
        except:
            pass
        self.show_roi2(r_color, roi_color, color_color)
        localtime = time.asctime(time.localtime(time.time()))
        if not self.cameraflag:
            if not (r_color or color_color):
                self.api_ctl2(pic_path)
                return
            value = [localtime, color_color, r_color, self.pic_source]
            # img_excel.excel_add(value)
            img_sql.sql(value[0], value[1], value[2], value[3])
        print(localtime, "|", color_color, r_color, "| ", self.pic_source)

    def pic2(self, pic_path):
        """对图像进行处理，处理完成后在主窗口左边显示要识别的图像，
        在右边展示--形状定位车牌位置--形状定位识别结果--颜色定位车牌位置--颜色定位识别结果--的信息"""
        self.apistr = None  # self.apistr字段表示的是，调用api进行车牌识别返回的车牌完整信息的字符串
        img_bgr = img_math.img_read(pic_path)  # 以uint8方式读取pic_path返回给img_bgr，即返回以uint8方式读取的彩色照片
        first_img, oldimg = self.predictor.img_first_pre(img_bgr)
        # 返回Otsu’s二值化边缘化的新图像和原图，需要两个变量接收
        # 其中first_img接收Otsu’s二值化边缘化的新图像
        if not self.cameraflag:
            # 这个if的作用相当于vedio_thread()，即传入的图像不是由相机拍摄的图像，将图像显示在主窗口左边
            # self.cameraflag置为0时，即传入的图像不是由相机拍摄的图像，将图像显示在主窗口左边
            self.imgtk = self.get_imgtk(img_bgr)
            self.image_ctl.configure(image=self.imgtk)  # 将图像显示在主窗口左边
        th2 = ThreadWithReturnValue(target=self.predictor.color_and_contour, args=(oldimg, oldimg, first_img))
        # 创建线程th2，通过run()方法回调color_and_contour()方法，并将args作为参数传入color_and_contour()方法中
        # color_and_contour()方法是通过"颜色定位车牌位置，颜色定位识别结果"的方法
        # th1.start()
        th2.start()
        r_color, roi_color, color_color = th2.join()

        try:
            Plate = HyperLPR_PlateRecogntion(img_bgr)
            # HyperLPR_PlateRecogntion是未完成的通过HyperLPR库进行车牌识别的更高版本
            # print(Plate[0][0])
            r_c = Plate[0][0]
            r_color = Plate[0][0]
        except:
            pass
        self.show_roi2(r_color, roi_color, color_color)
        localtime = time.asctime(time.localtime(time.time()))
        if not self.cameraflag:
            if not (r_color or color_color):
                self.api_ctl2(pic_path)
                return
            value = [localtime, color_color, r_color, self.pic_source]
            # img_excel.excel_add(value)
            img_sql.sql2(value[0], value[1], value[2], value[3])
        print(localtime, "|", color_color, r_color, "|", self.pic_source)

    def from_pic(self):
        """手动选择要识别的本地图像，之后调用pic()方法"""
        self.thread_run = False
        self.thread_run2 = False
        self.cameraflag = 0
        self.pic_path = askopenfilename(title="选择识别图片", filetypes=[("jpeg图片", "*.jpeg"), ("jpg图片", "*.jpg"), ("png图片", "*.png")])
        self.clean()
        self.pic_source = "本地文件：" + self.pic_path
        self.pic(self.pic_path)  # 将要识别的图像的路径传递给pic()方法

    def from_pic2(self):
        """进行路径批量识别，新建线程，调用pic_search方法"""
        self.pic_path3 = askdirectory(title="选择识别路径")
        self.get_img_list(self.pic_path3)  # 将路径传给get_img_list()方法，调用get_img_list()方法
        self.thread7 = threading.Thread(target=self.pic_search, args=(self,))
        # 通过run()方法回调pic_search()
        self.thread7.setDaemon(True)
        self.thread7.start()  # 启动线程thread7
        self.thread_run7 = True

    def get_img_list(self, images_path):
        """目的是为了生成一个存储图像路径的列表array_of_img，其中array_of_img是全局变量"""
        self.count = 0
        self.array_of_img = []
        for filename in os.listdir(images_path):
            # 遍历路径下的图像文件
            # print(filename)
            try:
                self.pilImage3 = Image.open(images_path + "/" + filename)
                self.array_of_img.append(images_path + "/" + filename)
                # 向列表中新增图像的完整路径
                self.count = self.count + 1
            except:
                pass
        print(self.array_of_img)

    def pic_search(self, self2):
        """遍历传入路径的所有图像，每一个图像都调用pic()方法进行处理"""
        self.thread_run7 = True
        print("开始批量识别")
        wait_time = time.time()
        while self.thread_run7:
            # 知道计数器count变为0，置参数thread_run7为False，关闭线程thread7
            while self.count:
                self.pic_path7 = self.array_of_img[self.count-1]

                if time.time()-wait_time > 2:
                    # print(self.pic_path7)
                    print("正在批量识别", self.count)
                    self.clean()
                    self.pic_source = "本地文件：" + self.pic_path7
                    try:
                        self.pic2(self.pic_path7)
                        # 调用pic()方法对图像进行处理，处理完成后在主窗口左边显示要识别的图像，
                        # 在右边展示–形状定位车牌位置–形状定位识别结果–颜色定位车牌位置–颜色定位识别结果–的信息
                    except:
                        pass
                    self.count = self.count - 1
                    wait_time = time.time()
            if self.count == 0:
                self.thread_run7 = False
                print("批量识别结束")
                return

    def vedio_thread(self):
        """将摄像头实时拍摄的视频显示在主窗口"""
        self.thread_run = True
        while self.thread_run:
            # 将摄像头实时拍摄的照片显示在主界面窗口
            _, self.img_bgr = self.camera.read()
            self.imgtk = self.get_imgtk(self.img_bgr)
            # 调用get_imgtk方法，传入img_bgr参数
            self.image_ctl.configure(image=self.imgtk)
            # 将摄像头拍摄的视频贴在标签image_ctl上
        print("run end")  # 当关闭程序时，thread_run置为False，执行下方print()语句

    def video_pic2(self):
        """把实时拍摄到的图像保存到本地，并调用pic()方法对传入的图像进行识别"""
        self.thread_run2 = True
        predict_time = time.time()
        while self.thread_run2:
            if self.cameraflag:
                if time.time() - predict_time > 2:
                    print("实时识别中self.cameraflag", self.cameraflag)
                    cv2.imwrite("tmp/timetest.jpg", self.img_bgr)
                    self.pic_path = "tmp/timetest.jpg"
                    self.pic(self.pic_path)  # 调用pic()方法对传入的图像进行识别
                    predict_time = time.time()
        print("run end")

    def video_pic(self):
        """拍摄照片，将照片保存到tmp/test.jpg，并调用pic()方法对图片进行处理"""
        if not self.thread_run:
            tkinter.messagebox.showinfo('提示', '请点击[打开摄像头]按钮！')
            return
        self.thread_run = False
        self.thread_run2 = False
        _, img_bgr = self.camera.read()
        cv2.imwrite("tmp/test.jpg", img_bgr)
        self.pic_path = "tmp/test.jpg"
        self.clean()
        self.pic(self.pic_path)
        print("video_pic")


    def api_ctl2(self, pic_path66):
        """自己定义的算法完全没识别出车牌信息，则调用api_ctl2进行车牌识别"""
        if self.thread_run:
            return
        self.thread_run = False
        self.thread_run2 = False
        colorstr, textstr = api_pic(pic_path66)  # 调用api进行车牌识别，返回颜色字段和车牌号字段，分别赋值给colorstr, textstr
        self.apistr = colorstr + textstr  # self.apistr字段的值包括车牌颜色和车牌号信息，即完整的车牌信息
        self.show_roi2(textstr, None, colorstr)
        localtime = time.asctime(time.localtime(time.time()))
        value = [localtime, textstr, colorstr, self.pic_source]
        print(localtime, textstr, colorstr, self.pic_source)
        img_sql.sql(value[0], value[1], value[2], value[3])

    def clean(self):
        """执行完一次操作后，清除信息，将所有参数、图片复原，实现重置窗口的效果"""
        if self.thread_run:
            self.cameraflag=0
            return
        self.thread_run = False
        self.thread_run2 = False
        # self.p1.set("")
        img_bgr3 = img_math.img_read("pic/hy.png")
        self.imgtk2 = self.get_imgtk(img_bgr3)
        self.image_ctl.configure(image=self.imgtk2)

        self.r_ct2.configure(text="")
        self.color_ct2.configure(text="", state='enable')

        self.pilImage3 = Image.open("pic/locate.png")
        w, h = self.pilImage3.size
        pil_image_resized = self.resize(w, h, self.pilImage3)
        self.tkImage3 = ImageTk.PhotoImage(image=pil_image_resized)
        self.roi_ct2.configure(image=self.tkImage3, state='enable')

def close_window():
    print("destroy")
    if surface.thread_run:
        surface.thread_run = False
        surface.thread.join(2.0)
    win.destroy()


if __name__ == '__main__':
    win = tk.Tk()

    surface = Surface(win)
    # close,退出输出destroy
    win.protocol('WM_DELETE_WINDOW', close_window)
    # 进入消息循环
    win.mainloop()
