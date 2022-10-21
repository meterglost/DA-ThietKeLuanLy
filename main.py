import numpy as np
import cv2 as cv

# ========== Read input ==========

img = cv.imread('inputImage.png')

# ========== Convert to grayscale ==========

grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imwrite('grayscale.png', grayscale)

# ========== Remove noise ==========

blured = cv.GaussianBlur(src=grayscale, ksize=(5, 5), sigmaX=0)
# cv.imwrite('blured.png', blured)

# ========== Canny Edge Detection ==========

edge = cv.Canny(blured, 100, 200, apertureSize = 3)
# cv.imwrite('edge.png', edge)

# ========== Mask ROI (Region Of Interest) ==========

vertices = np.array(
	[
		[
			(int(0.1 * img.shape[1]), int(0.9 * img.shape[0])),
			(int(0.4 * img.shape[1]), int(0.4 * img.shape[0])),
			(int(0.6 * img.shape[1]), int(0.4 * img.shape[0])),
			(int(0.9 * img.shape[1]), int(0.9 * img.shape[0]))
		]
	],
	dtype=np.int32,
)

mask = np.zeros_like(edge)
cv.fillPoly(mask, vertices, 255)
masked_edge = cv.bitwise_and(edge, mask)
# cv.imwrite('masked_edge.png', masked_edge)

# ========== Hough Line Detection ==========

lines = cv.HoughLinesP(masked_edge, 1, np.pi/180, 100, minLineLength=20, maxLineGap=5)
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv.line(img, (x1,y1), (x2,y2), (0,255,0), 2)
cv.polylines(img, [vertices.reshape((-1,1,2))], True, (0,0,255))
cv.imwrite('line.png', img)

cv.imshow('', img)
cv.waitKey()
