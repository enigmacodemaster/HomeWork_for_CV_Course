import cv2
import numpy as np

# 肤色检测，返回只保留肤色的图像
def get_skin_hsv(img):
	hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	back = np.zeros(img.shape, np.uint8)
	(h0, s0, v0) = cv2.split(hsv_img)
	(x, y) = h0.shape
	for i in range(x):
		for j in range(y):
			if (h0[i][j] > 0) and (h0[i][j] < 40) and (s0[i][j] > 60) and (s0[i][j] < 255) and (v0[i][j] > 70) and (v0[i][j] < 255):
				back[i][j] = img[i][j]
				img[i][j] = 0
	return back


def get_skin_yuv(img):
	ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
	(y, cr, cb) = cv2.split(ycrcb_img)
	(x, y) = cr.shape
	back = np.zeros(img.shape, np.uint8)
	for i in range(x):
		for j in range(y):
			if (cr[i][j] > 144.5) and (cr[i][j] < 200) and (cb[i][j] > 67) and (cb[i][j] < 136):
				back[i][j]=img[i][j]
				img[i][j]=0
	return back

# 提升对比度，采用直方图正规化
def imageHistNormalization(I):
	'''
	-- 直方图正规化
	将Imin和Imax之间的像素值映射到Omin到Omax也就是0-255的区间
	Imin到Imax的区间范围肯定是小于或等于Omin到Omax的
	正规化之后不用再进行截断操作了
	'''
	Imax = np.max(I)
	Imin = np.min(I)
	Omin, Omax = 0, 255
	a = float(Omax - Omin) / (Imax - Imin)
	b = Omin - a * Imin
	res = a * I + b
	res = res.astype(np.uint8)
	return res

# 提升对比度，线性变换
def imageHistLinearTransformation(I, a, b):
	'''
	输入：图像，对比度参数，亮度变化参数
	'''
	res = I * a + b
	res[res > 255] = 255
	res = np.round(res)
	res = res.astype(np.uint8)
	return res


'''
	预定义算子
'''
kernel_edge_y = np.ones((3,3), np.float32) # 边缘算子
kernel_edge_y[0,:]=[1,2,1]
kernel_edge_y[1,:]=[0,0,0]
kernel_edge_y[2,:]=[-1,-2,-1]

kernel_edge_x = np.ones((3,3), np.float32) # 边缘算子
kernel_edge_x[0,:]=[-1,0,1]
kernel_edge_x[1,:]=[-2,0,2]
kernel_edge_x[2,:]=[-1,0,1]

# 读取图像
# 从图像存放的位置读取
original_image = cv2.imread('week1_homework.png')
rows, cols, chn = original_image.shape

# 拷贝副本，深拷贝
img = original_image.copy()

# 获取面部
face = get_skin_yuv(img)

# 双边滤脸
face_bilateralFilter = cv2.bilateralFilter(face, 25, 42, 14)
# face = cv2.GaussianBlur(face, (3,3), 1)
img1 = face_bilateralFilter - face + 128
img2 = cv2.GaussianBlur(img1, (1,1), 1, 0)
# img2 = cv2.GaussianBlur(img1, (3,3), 1)
img3 = face + img2 * 2 - 255
img4 = cv2.addWeighted(face, 0.2, img3, 0.8, 0)
# img4 = cv2.medianBlur(img4, 3)
# face = cv2.bilateralFilter(face, 20, 40, 10)
img_final = cv2.add(img, img4)

# 边缘提取
# img_edge_y = cv2.filter2D(img_final, -1, kernel_edge_y)
# img_edge_x = cv2.filter2D(img_final, -1, kernel_edge_x)
# img_edge = img_edge_x + img_edge_y
# img_final = cv2.addWeighted(img_final, 0.9, img_edge, 0.1, 0)

# 幅值调整
img_final = imageHistNormalization(img_final)
img_final = imageHistLinearTransformation(img_final, 0.9, 10)

# cv2.imshow("RESULT", cv2.hconcat([original_image, img_final]))
cv2.imwrite('Beauty.png', img_final)
# 添加水印

cv2.waitKey(0)