from scipy.interpolate import interp2d, RectBivariateSpline
import numpy as np
import cv2

img = np.zeros((256, 256))
# img[50, 100] = 1
# img[60, 105] = 1
# img[70, 110] = 1

# x = [50, 65, 70, 75]
# y = [100, 105, 110, 115]
# z = [1, 1, 1, 1]
# # f = RectBivariateSpline(x, y, z)
# f = interp2d(np.linspace(0, 255, 256), np.linspace(0, 255, 256), img, kind='linear')

# for i in range(img.shape[0]):
#     for j in range(img.shape[1]):
#         img[i][j] = f(i,j)

A = [50,100]
B = [65,105]
d = np.sqrt(np.square(A[0]-B[0]) + np.square(A[1]-B[1]))
for i in range(int(d)):
    x_i = int(A[0] + i/d*(B[0]-A[0]))
    y_i = int(A[1] + i/d*(B[1]-A[1]))
    img[x_i, y_i] = 1

cv2.imshow("Image", img)
cv2.waitKey(-1)