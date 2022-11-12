import cv2 as cv
import numpy as np
from PIL import Image as im
from scipy.ndimage import interpolation as inter
from skimage.morphology import reconstruction
from pdf2image import convert_from_path
import PyPDF2
import pytesseract


def rescaleFrame(frame, scale=0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[1] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


def grayscale(image):
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)


def thresh(image):
    return cv.threshold(image, 130, 255, cv.THRESH_BINARY)[1]


def binarization(image):
    img = cv.medianBlur(image, 5)
    th2 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                               cv.THRESH_BINARY, 11, 2)
    th3 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv.THRESH_BINARY, 11, 2)
    return th3


def correct_skew(image, delta=1, limit=5):
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1, dtype=float)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
        return histogram, score

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv.getRotationMatrix2D(center, best_angle, 1.0)
    rotated = cv.warpAffine(image, M, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)

    return best_angle, rotated


def OtsuBbinarization(img):
    se = cv.getStructuringElement(cv.MORPH_RECT, (8, 8))
    bg = cv.morphologyEx(img, cv.MORPH_DILATE, se)
    out_gray = cv.divide(img, bg, scale=255)
    return cv.threshold(out_gray, 0, 255, cv.THRESH_OTSU)[1]


def opening(image):
    kernel = np.ones((2, 2), np.uint8)
    return cv.morphologyEx(image, cv.MORPH_OPEN, kernel)


def closing(image):
    # defining the kernel matrix
    kernel = np.ones((1, 1), np.uint8)
    closing = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel)
    return closing


def imfill(img):
    # https://stackoverflow.com/questions/36294025/python-equivalent-to-matlab-funciton-imfill-for-grayscale
    # Use the matlab reference Soille, P., Morphological Image Analysis: Principles and Applications, Springer-Verlag, 1999, pp. 208-209.
    #  6.3.7  Fillhole
    # The holes of a binary image correspond to the set of its regional minima which
    # are  not  connected  to  the image  border.  This  definition  holds  for  grey scale
    # images.  Hence,  filling  the holes of a  grey scale image comes down  to remove
    # all  minima  which  are  not  connected  to  the  image  border, or,  equivalently,
    # impose  the  set  of minima  which  are  connected  to  the  image  border.  The
    # marker image 1m  used  in  the morphological reconstruction by erosion is set
    # to the maximum image value except along its border where the values of the
    # original image are kept:

    seed = np.ones_like(img) * 255
    img[:, 0] = 0
    img[:, -1] = 0
    img[0, :] = 0
    img[-1, :] = 0
    seed[:, 0] = 0
    seed[:, -1] = 0
    seed[0, :] = 0
    seed[-1, :] = 0

    fill_img = reconstruction(seed, img, method='erosion')

    return fill_img


def erosion(image):
    # defining the kernel matrix
    kernel = np.ones((2, 2), np.uint8)
    return cv.erode(image, kernel, iterations=1)


def dilation(image):
    # defining the kernel matrix
    kernel = np.ones((1, 1), np.uint8)
    dilation = cv.dilate(image, kernel, iterations=1)
    return dilation


def noise_removal(image):
    kernel = np.ones((1, 1), np.uint8)
    image = cv.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv.erode(image, kernel, iterations=1)
    image = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel)
    image = cv.medianBlur(image, 3)
    return (image)


def thin_font(image):
    image = cv.bitwise_not(image)
    kernel = np.ones((2, 2), np.uint8)
    image = cv.erode(image, kernel, iterations=1)
    image = cv.bitwise_not(image)
    return (image)


def thick_font(image):
    image = cv.bitwise_not(image)
    kernel = np.ones((2, 2), np.uint8)
    image = cv.dilate(image, kernel, iterations=1)
    image = cv.bitwise_not(image)
    return (image)


def pdfToImgOCR(pdf):
    pages = convert_from_path(pdf, 500,
                              poppler_path='C:\\Users\\Arosh\\Desktop\\my_projects\\utils\\poppler-0.68.0\\bin')
    for page in pages:
        page.save('out.jpg', 'JPEG')
        ocr_result = pytesseract.image_to_string(page)
    return ocr_result


def pdfToText(pdf):
    reader = PyPDF2.PdfFileReader(pdf)
    page = reader.getPage(0)
    ocr_result = page.extractText()
    return ocr_result
