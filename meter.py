import datetime
import pandas as pd
from random import sample
import cv2
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import random
from sympy import *
import math
from yolo import YOLO
from PIL import Image
import os

yolov5_args = {
        "model_path"        : 'logs/best_epoch_weights.h5',
        "classes_path"      : 'model_data/voc_classes.txt',
        #---------------------------------------------------------------------#
        #   anchors_path代表先验框对应的txt文件，一般不修改。
        #   anchors_mask用于帮助代码找到对应的先验框，一般不修改。
        #---------------------------------------------------------------------#
        "anchors_path"      : 'model_data/yolo_anchors.txt',
        "anchors_mask"      : [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
        #---------------------------------------------------------------------#
        #   输入图片的大小，必须为32的倍数。
        #---------------------------------------------------------------------#
        "input_shape"       : [640, 640],
        #---------------------------------------------------------------------#
        #   所使用的YoloV5的版本。s、m、l、x
        #---------------------------------------------------------------------#
        "phi"               : 's',
        #---------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        #---------------------------------------------------------------------#
        "confidence"        : 0.5,
        #---------------------------------------------------------------------#
        #   非极大抑制所用到的nms_iou大小
        #---------------------------------------------------------------------#
        "nms_iou"           : 0.3,
        #---------------------------------------------------------------------#
        #   最大框的数量
        #---------------------------------------------------------------------#
        "max_boxes"         : 100,
        #---------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
        #---------------------------------------------------------------------#
        "letterbox_image"   : True,
    }

ImageDir = os.listdir("F:/yolov5-tf2-main/input")
def image(pic_path):
    if pic_path == 0:
        yolov5 = YOLO(**yolov5_args)
        for i in range(len(ImageDir)):
            ImagePath = "F:/yolov5-tf2-main/img/" + ImageDir[i]
            ImageName = "F:/yolov5-tf2-main/img/" + str(i) + ".jpg"
            img = Image.open(ImagePath)
            image, boxes, scores, classes = yolov5.detect_image(img)
            origin = np.asarray(image) #将数据转为矩阵
            image_bgr = cv2.cvtColor(np.asarray(origin), cv2.COLOR_RGB2BGR)#cv2下的色彩空间灰度化
            cv2.imwrite(ImageName, image_bgr)
    elif pic_path != 0:
        yolov5 = YOLO(**yolov5_args)
        img = Image.open(pic_path)#打开图片
        img2 = cv2.imread(pic_path)
        image, boxes, scores, classes = yolov5.detect_image(img)#yolov5检测
        origin = np.asarray(image)  # 将数据转为矩阵
        image_bgr = cv2.cvtColor(np.asarray(origin), cv2.COLOR_RGB2BGR)  # cv2下的色彩空间灰度化
        cv2.imwrite("F:/yolov5-tf2-main/img1/detected.jpg", image_bgr)
        #boxes内返回的是yolo预测出来的边框坐标，通过该坐标可以对原图像进行裁剪
        for j in range(boxes.shape[0]):
            # top, left, bottom, right = boxes[i]
            # 或者用下面这句等价
            top = boxes[0][0]
            left = boxes[0][1]
            bottom = boxes[0][2]
            right = boxes[0][3]
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5
            # 左上角点的坐标
            top = int(max(0, np.floor(top + 0.5).astype('int32')))
            left = int(max(0, np.floor(left + 0.5).astype('int32')))
            # 右下角点的坐标
            bottom = int(min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32')))
            right = int(min(np.shape(image)[1], np.floor(right + 0.5).astype('int32')))
            # 记录图片的高度与宽度
            a = bottom - top
            b = right - left
            print ('height', a)
            print ('with', b)
            croped_region = image_bgr[top:bottom, left:right]  # 先高后宽
            #cv2.imshow("cropimage", croped_region)
            # 将裁剪好的目标保存到本地
            j + 1
            cv2.imwrite("F:/yolov5-tf2-main/img1/cutted_img_"+str(j)+".jpg", croped_region)
            print('cropped successed')

            cv2.waitKey(0)
            cv2.destroyAllWindows()

class mential():
    def get_max_point(self, cnt):
        lmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
        rmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
        tmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
        bmost = tuple(cnt[cnt[:, :, 1].argmax()][0])
        pmost = [lmost, rmost, tmost, bmost]
        return pmost

    def distance(self, pmost, centerpoint):
        cx, cy = centerpoint
        distantion = []
        for point in pmost:
            dx, dy = point
            distantion.append((cx - dx) ** 2 + (cy - dy) ** 2)
        index_of_max = distantion.index((max(distantion)))
        return index_of_max

    def ds_ofpoint(self, a, b):
        x1, y1 = a
        x2, y2 = b
        distances = int(sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))
        return distances

    def findline(self, cp, lines):
        x, y = cp
        cntareas = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            aa = sqrt(min((x1 - x) ** 2 + (y1 - x) ** 2, (x2 - x) ** 2 + (y2 - x) ** 2))
            if (aa < 50):
                cntareas.append(line)
        print(cntareas)
        return cntareas


def angle(v1, v2):
    dx1 = v1[2] - v1[0]
    dy1 = v1[3] - v1[1]
    dx2 = v2[2] - v2[0]
    dy2 = v2[3] - v2[1]
    angle1 = math.atan2(dy1, dx1)
    angle1 = angle1 * 180 / math.pi

    print('angle1',angle1)
    angle2 = math.atan2(dy2, dx2)
    angle2 = angle2 * 180 / math.pi
    print('angle2',angle2)
    if angle1 * angle2 >= 0:
        if angle2<=0 and  angle2>=-90:
            included_angle =abs(angle1 - angle2)
        else:
            included_angle = 360-abs(angle1 - angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
        # if included_angle > 180:
        #     included_angle = 360 - included_angle
    return included_angle


def get_mode(arr):
    while 0 in arr:
        arr.remove(0)
    mode = []
    arr_appear = dict((a, arr.count(a)) for a in arr)  # 统计各个元素出现的次数
    if max(arr_appear.values()) == 1:  # 如果最大的出现为1
        arrs = np.array(arr)
        oo = np.median(arrs)
        return oo
    else:
        for k, v in arr_appear.items():  # 否则，出现次数最大的数字，就是众数
            if v == max(arr_appear.values()):
                mode.append(k)
    return mode


def remove_diff(deg):
    """
    :funtion :
    :param b:
    :param c:
    :return:
    """
    if (True):
        # new_nums = list(set(deg)) #剔除重复元素
        mean = np.mean(deg)
        var = np.var(deg)
        # print("原始数据共", len(deg), "个\n", deg)
        '''
        for i in range(len(deg)):
            print(deg[i],'→',(deg[i] - mean)/var)
            #另一个思路，先归一化，即标准正态化，再利用3σ原则剔除异常数据，反归一化即可还原数据
        '''
        # print("中位数:",np.median(deg))
        percentile = np.percentile(deg, (25, 50, 75), interpolation='midpoint')
        # print("分位数：", percentile)
        # 以下为箱线图的五个特征值
        Q1 = percentile[0]  # 上四分位数
        Q3 = percentile[2]  # 下四分位数
        IQR = Q3 - Q1  # 四分位距
        ulim = Q3 + 2.5 * IQR  # 上限 非异常范围内的最大值
        llim = Q1 - 1.5 * IQR  # 下限 非异常范围内的最小值

        new_deg = []
        uplim = []
        for i in range(len(deg)):
            if (llim < deg[i] and deg[i] < ulim):
                new_deg.append(deg[i])
        # print("清洗后数据共", len(new_deg), "个\n", new_deg)
    new_deg = np.mean(new_deg)

    return new_deg
    # 图表表达


flag = 0
p0 = 0


def markzero(path):
    img = cv2.imread(path)
    a = []

    def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
        global flag, p0
        if event == cv2.EVENT_LBUTTONDOWN:
            xy = "%d,%d" % (x, y)
            p0 = [x, y]
            print(x, y)
            cv2.circle(img, (x, y), 2, (0, 0, 255), thickness=-1)
            cv2.putText(img, '*0*', (x - 30, y), 1,
                        2.0, (0, 0, 0), thickness=2)
            # cv2.imshow("image", img)

        elif event == cv2.EVENT_LBUTTONUP:  # 鼠标左键fang
            cv2.destroyWindow("image")
            print(p0)

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)

    cv2.imshow('image', img)
    cv2.waitKey(5000)
    return p0
    # while (1):
    #     cv2.imshow("image", img)
    #     if cv2.waitKey(0)&0xFF>0:
    #     # if cv2.waitKey(500)|0xFF>0:
    #         print(flag)
    #         break


def cut_pic(path):
    """
    :param pyrMeanShiftFiltering(input, 10, 100) 均值滤波
    :param 霍夫概率圆检测
    :param mask操作提取圆
    :return: 半径，圆心位置

    """
    input = cv2.imread(path)
    dst = cv2.pyrMeanShiftFiltering(input, 10, 100)

    cimage = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(cimage, cv2.HOUGH_GRADIENT, 1, 80, param1=100, param2=20, minRadius=80, maxRadius=0)
    circles = np.uint16(np.around(circles))  # 把类型换成整数
    r_1 = circles[0, 0, 2]
    c_x = circles[0, 0, 0]
    c_y = circles[0, 0, 1]
    # print(input.shape[:2])
    circle = np.ones(input.shape, dtype="uint8")
    circle = circle * 255
    # print(circle)
    cv2.circle(circle, (c_x, c_y), int(r_1), 0, -1)
    # cv2.circle(circle, (c_x, c_y), int(r_1*0.65), (255,255,255), -1)
    # cv2.imshow("circle", circle)
    bitwiseOr = cv2.bitwise_or(input, circle)

    # cv2.circle(bitwiseOr, (c_x, c_y), 2, 0, -1)
    # cv2.imshow(pname+'_resize'+ptype, bitwiseOr)
    cv2.imwrite(pname + '_resize' + ptype, bitwiseOr)
    ninfo = [r_1, c_x, c_y]
    return ninfo


def linecontours(cp_info):
    """
    :funtion : 提取刻度线，指针
    :param a: 高斯滤波 GaussianBlur，自适应二值化adaptiveThreshold，闭运算
    :param b: 轮廓寻找 findContours，
    :return:kb,new_needleset
    """
    r_1, c_x, c_y = cp_info
    img = cv2.imread(pname + '_resize' + ptype)

    cv2.circle(img, (c_x, c_y), 20, (23, 28, 28), -1)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('dds', img)
    # ret, binary = cv2.threshold(~gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    binary = cv2.adaptiveThreshold(~gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -10)
    # cv2.circle(binary, (c_x, c_y), int(r_1*0.5), (0, 0, 0),5)
    # 闭运算
    # kernel = np.ones((3, 3), np.uint8)
    # dilation = cv2.dilate(binary, kernel, iterations=1)
    # kernel2 = np.ones((3, 3), np.uint8)
    # erosion = cv2.erode(dilation, kernel2, iterations=1)

    # ************************
    # cv2.imshow('dds', binary)

    # aa, contours, hier = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours, hier = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  


    cntset = []  # 刻度线轮廓集合
    cntareas = []  # 刻度线面积集合
    needlecnt = []  # 指针轮廓集合
    needleareas = []  # 指针面积集合
    ca = (c_x, c_y)
    incircle = [r_1 * 0.7, r_1 * 0.9]

    # incircle = [r_1 * 0.1, r_1 * 1]
    # cv2.drawContours(img, contours, -1, (255, 90, 60), 2)
    localtion = []
    for xx in contours:
        rect = cv2.minAreaRect(xx)
        # print(rect)
        a, b, c = rect
        w, h = b
        w = int(w)
        h = int(h)
        ''' 满足条件:“长宽比例”，“面积”'''
        if h == 0 or w == 0:
            pass
        else:
            dis = mential.ds_ofpoint(self=0, a=ca, b=a)

            if (incircle[0] < dis and incircle[1] > dis):
                localtion.append(dis)
                if h / w > 4 or w / h > 4:
                    cntset.append(xx)
                    cntareas.append(w * h)
            else:
                if w > r_1 / 2 or h > r_1 / 2:
                    needlecnt.append(xx)
                    needleareas.append(w * h)
    cntareas = np.array(cntareas)
    nss = remove_diff(cntareas)  # 中位数，上限区
    new_cntset = []
    # 面积
    for i, xx in enumerate(cntset):
        if (cntareas[i] <= nss * 1.5 and cntareas[i] >= nss * 0.8):
            new_cntset.append(xx)
    kb = []  # 拟合线集合
    for xx in new_cntset:
        rect = cv2.minAreaRect(xx)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.polylines(img, [box], True, (0, 255, 0), 1)  # pic
        output = cv2.fitLine(xx, 2, 0, 0.001, 0.001)
        k = output[1] / output[0]
        k = round(k[0], 2)
        b = output[3] - k * output[2]
        b = round(b[0], 2)
        x1 = 1
        x2 = gray.shape[0]
        y1 = int(k * x1 + b)
        y2 = int(k * x2 + b)
        # cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        kb.append([k, b])  # 求中心点的点集[k,b]

    ############################################################
    r = np.mean(localtion)
    mask = np.zeros(img.shape[0:2], np.uint8)
    # for cnt in needlecnt:
    #     cv2.fillConvexPoly(mask,cnt , 255)
    mask = cv2.drawContours(mask, needlecnt, -1, (255, 255, 255), -1)  # 生成掩膜
    # cv2.imshow('da', mask)

    cv2.imwrite(pname + '_scale' + ptype, img)
    cv2.imwrite(pname + '_needle' + ptype, mask)
    return kb, r, mask


def needle(path,img, r, cx, cy,x0,y0):
    oimg =cv2.imread(path)
    # circle = np.ones(img.shape, dtype="uint8")
    # circle = circle * 255
    circle = np.zeros(img.shape, dtype="uint8")
    cv2.circle(circle, (cx, cy), int(r), 255, -1)
    mask = cv2.bitwise_and(img, circle)
    # cv2.imshow('m', mask)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    # erosion = cv2.erode(mask, kernel, iterations=1)
    # cv2.imshow('1big', mask)

    lines = cv2.HoughLinesP(mask, 1, np.pi / 180, 100, minLineLength=int(r / 2), maxLineGap=2)
    nmask = np.zeros(img.shape, np.uint8)
    # lines = mential.findline(self=0, cp=[x, y], lines=lines)
    # print('lens', len(lines))
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(nmask, (x1, y1), (x2, y2), 100, 1, cv2.LINE_AA)

    x1, y1, x2, y2 = lines[0][0]
    d1 = (x1 - cx) ** 2 + (y1 - cy) ** 2
    d2 = (x2 - cx) ** 2 + (y2 - cy) ** 2
    if d1 > d2:
        axit = [x1, y1]
    else:
        axit = [x2, y2]
    nmask = cv2.erode(nmask, kernel, iterations=1)

    # cv2.imshow('2new', nmask)
    # aa, cnts, hier = cv2.findContours(nmask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts, hier = cv2.findContours(nmask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    areass = [cv2.contourArea(x) for x in cnts]
    # print(len(areass))
    i = areass.index(max(areass))
    # print('contours[i]',contours[i])
    # cv2.drawContours(img, contours[i], -1, (10,20,250), 1)
    # cv2.imshow('need_next', img)
    cnt = cnts[i]
    output = cv2.fitLine(cnt, 2, 0, 0.001, 0.001)
    k = output[1] / output[0]
    k = round(k[0], 2)
    b = output[3] - k * output[2]
    b = round(b[0], 2)
    x1 = cx
    x2 = axit[0]
    y1 = int(k * x1 + b)
    y2 = int(k * x2 + b)
    cv2.line(oimg, (x1, y1), (x2, y2), (0, 23, 255), 1, cv2.LINE_AA)
    cv2.line(oimg, (x1, y1), (x0,y0), (0, 23, 255), 1, cv2.LINE_AA)
    # cv2.imshow('msss', oimg)
    print(pname +'_fin'+ ptype)
    cv2.imwrite(pname +'_fin'+ ptype,oimg)
    return x1, y1, x2, y2


def findpoint(kb,path):
    img = cv2.imread(path)
    w, h, c = img.shape
    point_list = []
    if len(kb) > 2:
        # print(len(kb))
        random.shuffle(kb)
        lkb = int(len(kb) / 2)
        kb1 = kb[0:lkb]
        kb2 = kb[lkb:(2 * lkb)]
        print('len', len(kb1), len(kb2))
        kb1sample = sample(kb1, int(len(kb1) / 2))
        kb2sample = sample(kb2, int(len(kb2) / 2))
    else:
        kb1sample = kb[0]
        kb2sample = kb[1]

    for i, wx in enumerate(kb1sample):
        # for wy in kb2:
        for wy in kb2sample:
            k1, b1 = wx
            k2, b2 = wy
            # print('kkkbbbb',k1[0],b1[0],k2[0],b2[0])
            # k1-->[123]
            try:
                if (b2 - b1) == 0:
                    b2 = b2 - 0.1
                if (k1 - k2) == 0:
                    k1 = k1 - 0.1
                x = (b2 - b1) / (k1 - k2)
                y = k1 * x + b1
                x = int(round(x))
                y = int(round(y))
            except:
                x = (b2 - b1 - 0.01) / (k1 - k2 + 0.01)
                y = k1 * x + b1
                x = int(round(x))
                y = int(round(y))
            # x,y=solve_point(k1, b1, k2, b2)
            if x < 0 or y < 0 or x > w or y > h:
                break
            point_list.append([x, y])
            cv2.circle(img, (x, y), 2, (122, 22, 0), 2)
    # print('point_list',point_list)
    if len(kb) > 2:
        # cv2.imshow(pname+'_pointset',img)
        cv2.imwrite(pname + '_pointset' + ptype, img)
    return point_list


def countpoint(pointlist,path):
    # pointlist=[[1,2],[36,78],[36,77],[300,300],[300,300]]
    img = cv2.imread(path, 0)
    h, w = img.shape
    pic_list = np.zeros((h, w))
    for point in pointlist:
        # print('point',point)
        x, y = point
        if x < w and y < h:
            pic_list[y][x] += 1
    # print(pic_list)
    cc = np.where(pic_list == np.max(pic_list))
    # print(cc,len(cc))
    y, x = cc
    cc = (x[0], y[0])
    cv2.circle(img, cc, 2, (32, 3, 240), 3)
    # cv2.imshow(pname + '_center_point', img)
    cv2.imwrite(pname + '_center_point' + ptype, img)
    return cc


import datetime

pname, ptype=0,0
def decter(path,outpath,opoint):
    x0=opoint[0]
    y0=opoint[1]
    global  pname, ptype
    pname, ptype = path.split('.')
    ptype = '.' + ptype
    pname=outpath+'/'+pname.split('/')[-1]
    print(pname)

    # if(os.path.exists(pname)):
    #     print('exit')
    start = datetime.datetime.now()
    ninfo = cut_pic(path)  # 2.截取表盘
    kb, r, mask = linecontours(ninfo)
    point_list = findpoint(kb,path)
    cx, cy = countpoint(point_list,path)
    da, db, dc, de = needle(path,mask, r, cx, cy,x0, y0)
    # da,db,dc,de=needle_line(lines,new_needleset,cx,cy)
    # print(da,db,dc,de)
    distinguish = 100 / 360

    OZ = [da, db, x0, y0]
    OP = [da, db, dc, de]
    ang1 = angle(OZ, OP)
    output=ang1 * distinguish
    print("AB和CD的夹角")
    print()
    # output=str(output)
    end = datetime.datetime.now()
    print(end - start)
    cv2.waitKey(1)
    # cv2.destroyAllWindows()
    return output

markzero(image('F:/yolov5-tf2-main/input/5.jpg'))
ang1 = decter('F:/yolov5-tf2-main/input/5.jpg')
print(ang1)


