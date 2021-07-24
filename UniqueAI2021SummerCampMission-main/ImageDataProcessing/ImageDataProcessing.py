# 导入要用的库
import cv2
import numpy as np
import math
from skimage import data, exposure, img_as_float
from PIL import Image as p
import matplotlib.pyplot as plt


class ImageDataProcessing:
    """
        图像数据处理
        传入地址list，从前往后一次对图片进行不同处理
    """
    def __init__(self, path):
        self.path = path

    def rgb2hsl(self):
        # 实现rgb到hsl的转换功能
        # 待填, 对self.path[0]操作
        # 请分别打印self.path[0]的rgb和hsl像素值
        img = cv2.imread(self.path[0])
        h, w, d = img.shape
        hls_array = np.empty(shape=(h, w, 3), dtype=float)
        print(h, w)
        for i in range(0, int(h)):
            for j in range(0, int(w)):
                r = img[i][j][0] / 255
                g = img[i][j][1] / 255
                b = img[i][j][2] / 255
                Max = max(r, g, b)
                Min = min(r, g, b)
                h = s = l = (Max + Min) / 2
                if Max == Min:
                    h = s = 0
                else:
                    d = Max - Min
                    s = d / (2 - Max - Min) if l > 0.5 else d / (Max + Min)
                    if Max == r:
                        h = (g - b) / d + (6 if g < b else 0)
                    elif Max == g:
                        h = (b - r) / d + 2
                    else:
                        h = (r - g) / d + 4
                    h *= 60
                    if h >= 360:
                        h -= 360
                    elif h < 0:
                        h += 360
                hls_array[i][j][0] = h
                hls_array[i][j][1] = s
                hls_array[i][j][2] = l

        print("RGB:\n",img)
        print("HSL:\n",hls_array)
        pass

    def vague(self):
        # 实现将图片变模糊功能
        # 待填，对self.path[1]进行操作
        # 请分别打印self.path[1]的模糊前和模糊后的图像
        img = cv2.imread(self.path[1])
        sizepic = [0, 0]
        h, w, d = img.shape
        newimg = np.empty((h, w, d))    #新建一张图片
        PI = 3.14159265
        r = 1
        rd = 4
        sizepic[0] = h
        sizepic[1] = w
        nr = np.zeros((sizepic[0], sizepic[1]))
        ng = np.zeros((sizepic[0], sizepic[1]))
        nb = np.zeros((sizepic[0], sizepic[1]))
        for i in range(0, sizepic[0]):  #读取原有图片的RGB
            for j in range(0, sizepic[1]):
                nr[i][j] = img[i][j][0]
                ng[i][j] = img[i][j][1]
                nb[i][j] = img[i][j][2]

        summat = 0
        ma = np.empty((2 * r + 1, 2 * r + 1))
        for i in range(0, 2 * r + 1):
            for j in range(0, 2 * r + 1):
                gaussp = (1 / (2 * PI * (r ** 2))) * math.e ** (-((i - r) ** 2 + (j - r) ** 2) / (2 * (r ** 2)))
                ma[i][j] = gaussp
                summat += gaussp

        for i in range(0, 2 * r + 1):
            for j in range(0, 2 * r + 1):
                ma[i][j] = ma[i][j] / summat    #高斯函数矩阵
        newr = np.empty((sizepic[0], sizepic[1]))
        newg = np.empty((sizepic[0], sizepic[1]))
        newb = np.empty((sizepic[0], sizepic[1]))
        for i in range(r + 1, sizepic[0] - r):
            for j in range(r + 1, sizepic[1] - r):
                o = 0
                for x in range(i - r, i + r + 1):
                    p = 0
                    for y in range(j - r, j + r + 1):
                        # print("x{},y{},o{},p{}".format(x,y,o,p))
                        newr[i][j] += nr[x][y] * ma[o][p]
                        newg[i][j] += ng[x][y] * ma[o][p]
                        newb[i][j] += nb[x][y] * ma[o][p]
                        p += 1
                    o += 1
        for i in range(rd + 1, sizepic[0] - rd + 1):
            for j in range(rd + 1, sizepic[1] - rd + 1):#输出新图片
                newimg[i][j][0] = newr[i][j]
                newimg[i][j][1] = newg[i][j]
                newimg[i][j][2] = newb[i][j]
        cv2.imwrite('vague.jpg', newimg)
        pass

    def noise_reduction(self):
        # 实现将图片降噪功能
        # 待填，对self.path[2]进行操作
        # 请分别打印self.path[2]的降噪前和降噪后的图像
        img = cv2.imread(self.path[2])
        cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        cv2.imwrite('noise_reduction.jpg', img)
        pass

    def edge_extraction(self):
        # 实现边缘提取功能
        # 待填，对self.path[3]进行操作
        # 请分别打印self.path[3]的边缘提取前后的图像
        img = cv2.imread(self.path[3], cv2.IMREAD_GRAYSCALE) #robot算子对灰度图卷积
        h, w = img.shape
        new_image = np.zeros((h, w))
        '''
        L_sunnzi = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]]) #拉普拉斯算子
        # L_sunnzi = np.array([[1,1,1],[1,-8,1],[1,1,1]])
        for i in range(h-2):
            for j in range(w-2):
              new_image[i+1, j+1] = abs(np.sum(img[i:i+3, j:j+3] * L_sunnzi))
        '''
        r_sunnzi = [[-1, -1], [1, 1]]
        for x in range(h):
            for y in range(w):
                if (y + 2 <= w) and (x + 2 <= h):
                    imgChild = img[x:x + 2, y:y + 2]
                    list_robert = r_sunnzi * imgChild
                    new_image[x, y] = abs(list_robert.sum())
        cv2.imwrite("edge_extraction.jpg", new_image)
        pass

    def brightness_adjustment(self):#调用了skimage的库
        # 实现亮度调整功能
        # 待填，对self.path[4]进行操作
        # 请分别打印self.path[4]的原图，变亮后图像，变暗后图像
        img = cv2.imread(self.path[4])
        gam1 = exposure.adjust_gamma(img, 2)  # 调暗
        gam2 = exposure.adjust_gamma(img, 0.5) #调亮
        cv2.imwrite("bright_adjustment.jpg", gam2)
        cv2.imwrite("dark_adjustment.jpg", gam1)
        pass

    def rotate(self):
        # 实现旋转功能
        # 待填，对self.path[5]进行操作
        # 请分别打印self.path[5]的原图，旋转任意角度后图像
        angle = 10
        img = p.open(self.path[5])
        img = img.rotate(angle)
        plt.imshow(img)
        plt.show()

        #自己实现不了，用矩阵算出来的图片信息丢失严重
        '''
        anglePi = angle * math.pi / 180.0
        cosA = math.cos(anglePi)
        sinA = math.sin(anglePi)

        
        for i in range(h):
            for j in range(w):
                x = int(cosA*i-sinA*j-0.5*w*cosA+0.5*h*sinA+0.5*w)
                y = int(sinA*i+cosA*j-0.5*w*sinA-0.5*h*cosA+0.5*h)
                if x>-1 and x<h and y>-1 and y<w:
                    newimg[x,y] = img[i,j]
        '''
        '''
        for i in range(h):
            for j in range(w):
                anglePi = angle * math.pi / 180.0
                x = int(math.cos(anglePi) * i + math.sin(anglePi) * j)
                y = int(-math.sin(anglePi) * i + math.cos(anglePi) * j)
                if x > -1 and x < h and y > -1 and y < w:
                    newimg[x, y] = img[i, j]
        '''

        pass

    def flip_horizontally(self):
        # 实现水平翻转功能
        # 待填，对self.path[6]进行操作
        # 请分别打印self.path[6]的原图，水平翻转后图像
        img = cv2.imread(self.path[6])
        img = img[:, ::-1, :]
        cv2.imwrite('flip_horizontally.jpg',img)
        pass

    def cutting(self):
        # 实现裁切功能
        # 待填，对self.path[7]进行操作
        # 请分别打印self.path[7]的原图，裁切后图像
        img = cv2.imread(self.path[6])
        x = 100
        y = 100
        img = img[x:, y:, :]
        cv2.imwrite('cutting.jpg', img)
        pass

    def resize(self):
        # 实现调整大小功能
        # 待填，对self.path[8]进行操作
        # 请分别打印self.path[8]的原图，调整任意大小后图像

        img = cv2.imread(self.path[8])
        target_size = [100, 100, 3]
        bilinear_img = np.zeros(shape=(target_size[0], target_size[1], target_size[2]), dtype=np.uint8) #双线性插值
        for i in range(0, target_size[0]):
            for j in range(0, target_size[1]):
                row = (i / target_size[0]) * img.shape[0]
                col = (j / target_size[1]) * img.shape[1]
                row_int = int(row)
                col_int = int(col)
                u = row - row_int
                v = col - col_int
                if row_int == img.shape[0] - 1 or col_int == img.shape[1] - 1:
                    row_int -= 1
                    col_int -= 1
                bilinear_img[i][j] = (1 - u) * (1 - v) * img[row_int][col_int] + (1 - u) * v * img[row_int][
                    col_int + 1] + u * (1 - v) * img[row_int + 1][col_int] + u * v * img[row_int + 1][col_int + 1]
        cv2.imwrite('resize.jpg', bilinear_img)
        pass

    def normalization(self):
        # 实现归一化功能
        # 待填，对self.path[9]进行操作
        # 请分别打印self.path[9]的原图，归一化后图像
        img = cv2.imread(self.path[9])
        new_img = (img - np.min(img)) / (np.max(img) - np.min(img))
        cv2.imwrite('normalization.jpg',new_img)
        pass

    def fit(self):
        self.rgb2hsl()
        self.vague()
        self.noise_reduction()
        self.edge_extraction()
        self.brightness_adjustment()
        self.rotate()
        self.flip_horizontally()
        self.cutting()
        self.resize()
        self.normalization()


if __name__ == '__main__':
    ImageDataProcessing(["pics/0.jpg", "pics/3.jpg", "pics/2.jpg", "pics/8.jpg", "pics/9.jpg", "pics/1.jpg",
                         "pics/7.jpg", "pics/6.jpg", "pics/5.jpg", "pics/4.jpg"]).fit()
