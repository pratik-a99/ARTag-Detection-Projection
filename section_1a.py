import cv2
import numpy as np

img = cv2.imread('test.png', 0)
th, img_bin = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)
# cv2.imshow("orig", img_bin)
dft = cv2.dft(np.float32(img_bin), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

magnitude_spectrum = np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

rows, cols = img_bin.shape
crow, ccol = int(rows / 2), int(cols / 2)  # center

# Circular HPF mask, center circle is 0, remaining all ones

mask = np.ones((rows, cols, 2), np.uint8)
r = 180
center = [crow, ccol]
x, y = np.ogrid[:rows, :cols]
mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
mask[mask_area] = 0

# apply mask and inverse DFT
fshift = dft_shift * mask

fshift_mask_mag = 2000 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))

f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])


img_back = cv2.GaussianBlur(img_back, (5, 5), 0)

kernel = np.ones((3, 3), np.uint8)
img_back = cv2.erode(img_back, kernel, iterations=2)
img_back = cv2.GaussianBlur(img_back, (7, 7), 0)

# kernel = np.ones((3, 3), np.uint8)
# img_back = cv2.dilate(img_back, kernel, iterations=2)
# kernel = np.ones((3, 3), np.uint8)
# img_back = cv2.erode(img_back, kernel)


cv2.imshow('Corners', img_back)
# cv2.imwrite("fft_result.png", img_back)

# if cv2.waitKey(0) & 0xff == 27:
#     cv2.destroyAllWindows()

cv2.waitKey()
