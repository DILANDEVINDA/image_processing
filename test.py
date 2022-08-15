import cv2
import cv2 as cv
import numpy as np
from functions import rescaleFrame, grayscale, opening, correct_skew, thresh, erosion, closing, OtsuBbinarization, imfill, dilation, correct_skew
from matplotlib import pyplot as plt
import pytesseract
from PIL import Image

path = './assets/test3.jpg'
img = cv.imread(path)

pytesseract.pytesseract.tesseract_cmd =r'C:\Program Files\Tesseract-OCR\tesseract.exe'

#cv.imshow("original", img)
bill = rescaleFrame(img, 0.5)
cv.imshow("Rescaled", img)

grayscale = grayscale(img)
#cv.imshow("Grayscale", grayscale)

kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
sharpen = cv2.filter2D(src=grayscale, ddepth=-1, kernel=kernel)
cv.imshow("Sharpen", sharpen)

#thresh, im_bw = cv.threshold(grayscale, 130, 255, cv.THRESH_BINARY)
#cv.imshow("Thresh", im_bw)

#erosion = erosion(opening)
#cv.imshow("Erosion", erosion)


#closing = closing(otsu)
#cv.imshow("Closing", closing)

denoise = cv2.fastNlMeansDenoising(sharpen, None,  30, 7, 21)
cv.imshow("Denoising", denoise)

#blur = cv.GaussianBlur(denoise, (3, 3), 0)
#cv.imshow("Blur", blur)

# otsu = OtsuBbinarization(cv.blur(denoise, (2, 2)))

blur = cv.GaussianBlur(denoise, (3, 3), 0)
otsu = OtsuBbinarization(blur)
cv.imshow("otsu", otsu)

# cv.imshow("1", cv2.fastNlMeansDenoising(closing, None, 30, 7, 21))

opening = opening(otsu)
cv.imshow("Opening image1", opening)

erosion = erosion(opening)
#cv.imshow("erosion image1", erosion)

blur = cv.GaussianBlur(opening, (1, 1), 0)
otsu = OtsuBbinarization(blur)
#cv.imshow("otsu2", otsu)

#blur = cv.GaussianBlur(opening, (3, 3), 0)

#otsu = OtsuBbinarization(blur)
#cv.imshow("otsu1", otsu)



#erosion = erosion(otsu)
#cv.imshow("Erosion", erosion)





im = Image.fromarray(opening)
ocr_result = pytesseract.image_to_string(im, lang='sin')
print(ocr_result)

cv.waitKey(0)
