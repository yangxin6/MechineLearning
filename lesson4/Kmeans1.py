import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets._samples_generator import make_blobs
from sklearn.cluster import KMeans
import cv2


def segmentation(path):
    src = cv2.imread(path)
    img = src.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    print(thresh.shape)

    # 噪声去除
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    # 确定背景区域
    sure_bg = cv2.dilate(opening, kernel, iterations=5)  # 膨胀
    # 寻找前景区域-对象分离
    # separate分离系数，取值范围0.1-1
    separate = 0.4
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
    ret, sure_fg = cv2.threshold(dist_transform, separate * dist_transform.max(), 255, 0)  # sure_fg为分离对象的图像
    # 找到未知区域
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 类别标记
    ret, markers = cv2.connectedComponents(sure_fg)
    # 为所有的标记加1，保证背景是0而不是1
    markers = markers + 1
    # 现在让所有的未知区域为0
    markers[unknown == 255] = 0

    markers = cv2.watershed(img, markers)
    img[markers > 1] = [255, 0, 0]

    # dist_transform = cv2.normalize(dist_transform, 0, 0.8, cv2.NORM_MINMAX) * 10

    # cv2.imshow("dist_transform", dist_transform)
    #
    # cv2.imshow("img", img)
    # cv2.waitKey()
    return img, src


def calMainLabel(label, kernel, point):
    # for i in range(kernel[0]):
    #     img2[point[0] + i][point[1]] = colors[1]
    # for j in range(kernel[1]):
    #     img2[point[0]][point[1] + j] = colors[1]
    # for i in range(kernel[0]):
    #     img2[point[0] + i][point[1] + kernel[1]] = colors[1]
    # for j in range(kernel[1]):
    #     img2[point[0] + kernel[0]][point[1] + j] = colors[1]
    # cv2.imshow("rect", img)
    # cv2.waitKey()
    kernelTotalLabel = {}
    for i in range(kernel[0]):
        for j in range(kernel[1]):
            if label[point[0] + i][point[1] + j] not in kernelTotalLabel:
                kernelTotalLabel[label[point[0] + i][point[1] + j]] = 1
            else:
                kernelTotalLabel[label[point[0] + i][point[1] + j]] += 1
    return max(kernelTotalLabel, key=kernelTotalLabel.get)

# 功能：加载图像数据
# 输入：
#     path: 文件路径
# 输出：
#     n * 3 的矩阵
def kMeams(img, src, k):
    # img = cv2.imread(path)
    height = img.shape[0]
    width = img.shape[1]
    # 为不同的类分配不同的颜色
    colors = [[0, 0, 0],
              [255, 0, 0],  # 红
              [255, 255, 0],  # 黄
              [0, 0, 255],  # 蓝
              [0, 255, 0],  # 绿
              [0, 255, 255],  # 浅蓝
              [255, 0, 255],
              [255, 170, 255],
              [170, 170, 255],
              [255, 255, 255]]  # 紫

    ## 第二次 Kmeans 聚类
    img1 = img.copy()
    img2 = src.copy()
    dataset1 = img1.transpose(2, 0, 1).reshape(3, -1).transpose(1, 0)
    kMeans = KMeans(n_clusters=k)
    kMeans.fit(dataset1)
    label = kMeans.labels_.reshape((height, width))

    for i in range(height):
        for j in range(width):
            for c in range(k):
                if label[i][j] == c:
                    img1[i][j] = colors[c % len(colors)]
    # cv2.imshow("img1", img1)
    cv2.imwrite("./k=%s-cluster.jpg" % k, img1)
    # 取斑点中聚类最多的类
    maxLabelKey = []
    kernel = [20, 10]  # H W
    point = [289, 359]  # H W
    maxLabelKey.append(calMainLabel(label, kernel, point))
    #
    # kernel2 = [9, 12]  # H W
    # point2 = [165, 121]  # H W
    # maxLabelKey.append(calMainLabel(label, kernel2, point2))

    for i in range(height):
        for j in range(width):
            if label[i][j] in maxLabelKey:
                img2[i][j] = colors[label[i][j] % len(colors)]
            # for c in range(k):
            #     if label[i][j] == c:
            #         img1[i][j] = colors[c]
    # cv2.imshow("img2", img2)
    cv2.imwrite("./k=%s-result.jpg" % k, img2)
    # cv2.waitKey()



if __name__ == '__main__':
    # k值
    k = 13
    img, src = segmentation('tomato_sick.jpeg')
    kMeams(img, src, k)