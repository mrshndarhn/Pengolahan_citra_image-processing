import cv2
import numpy as np
from PIL import Image


# Melakukan transformasi warping affine pada gambar
def warpaffine(image):
    try:
        rows, cols, ch = image.shape
        pts1 = np.float32([[50, 50],
                           [200, 50],
                           [50, 200]])
        pts2 = np.float32([[50, 100],
                           [200, 50],
                           [150, 200]])
        points = cv2.getAffineTransform(pts1, pts2)
        img = cv2.wrapAffine(image, points, (cols, rows))
        img_conv = Image.fromarray(img)
        return img_conv
    except Exception as e:
        print(f"Error: {e}. Terjadi kesalahan saat melakukan transformasi warping affine pada gambar.")

# Melakukan deteksi tepi pada gambar
def edge_detection(image):
    try:
        edges = cv2.Canny(image, 40, 80, L2gradient=True)
        img_conv = Image.fromarray(edges)
        return img_conv
    except Exception as e:
        print(f"Error: {e}. Terjadi kesalahan saat melakukan deteksi tepi.")

# Melakukan konversi ke skala abu-abu
def gray_scale(image):
    try:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_conv = Image.fromarray(gray_image)
        return img_conv
    except Exception as e:
        print(f"Error: {e}. Terjadi kesalahan saat melakukan konversi ke skala abu-abu.")

# Melakukan transformasi negatif pada gambar
def negative_transformation(image):
    try:
        height, width, _ = image.shape
        for i in range(0, height - 1):
            for j in range(0, width - 1):
                try:
                    pixel = image[i, j]
                    pixel[0] = 255 - pixel[0]
                    pixel[1] = 255 - pixel[1]
                    pixel[2] = 255 - pixel[2]
                except IndexError as e:
                    print(f"Error: {e}. Pastikan ukuran gambar cukup besar.")
        return image
    except Exception as e:
        print(f"Error: {e}. Terjadi kesalahan saat melakukan transformasi negatif.")

# Melakukan blur Gaussian pada gambar
def Gaussian_Blur(image):
    try:
        blur = cv2.GaussianBlur(image, (5, 5), cv2.BORDER_DEFAULT)
        img_conv = Image.fromarray(blur)
        return img_conv
    except Exception as e:
        print(f"Error: {e}. Terjadi kesalahan saat melakukan blur Gaussian.")

# Mengurangi noise pada gambar
def reduce_noise(image):
    try:
        # ALGORITMA PENGURANGAN NOISE NON-LOCAL MEANS
        noiseless_image_colored = cv2.fastNlMeansDenoisingColored(image, None, 20, 20, 7, 21)
        img_conv = Image.fromarray(noiseless_image_colored)
        return img_conv
    except Exception as e:
        print(f"Error: {e}. Terjadi kesalahan saat mengurangi noise pada gambar.")

# Melakukan sharpening pada gambar
def sharp_image(image):
    try:
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]], dtype=np.float32)
        image_sharp = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
        img_conv = Image.fromarray(image_sharp)
        return img_conv
    except Exception as e:
        print(f"Error: {e}. Terjadi kesalahan saat melakukan sharpening pada gambar.")

#------------Embosis-------------------

def emboss(image):
    try:
        kernel = np.array([[-2, -1, 0],
                           [-1, 1, 1],
                           [0, 1, 2]])
        image_emboss = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
        img_conv = Image.fromarray(image_emboss)
        return img_conv
    except Exception as e:
        print(f"Error: {e}. Terjadi kesalahan saat melakukan emboss pada gambar.")

#------------High-Boost-------------------

def high_boost(image, alpha=2.0):
    try:
        # High-boost filtering = (alpha - 1) * original_image - alpha * low-pass_filtered_image
        low_pass = cv2.GaussianBlur(image.astype(np.float64), (5, 5), 0)
        high_boost_image = alpha * image.astype(np.float64) - (alpha - 1) * low_pass
        # Konversi kembali ke tipe data uint8 untuk kompatibilitas dengan Image.fromarray
        high_boost_image = np.clip(high_boost_image, 0, 255).astype(np.uint8)
        img_conv = Image.fromarray(high_boost_image)
        return img_conv
    except Exception as e:
        print(f"Error: {e}. Terjadi kesalahan saat melakukan high-boost filtering pada gambar.")

#------------Low-Pass-------------------

def low_pass(image):
    try:
        low_pass_image = cv2.GaussianBlur(image, (5, 5), 0)
        img_conv = Image.fromarray(low_pass_image)
        return img_conv
    except Exception as e:
        print(f"Error: {e}. Terjadi kesalahan saat melakukan low-pass filtering pada gambar.")

# Rotasi
def rotation(image, angle):
    try:
        rows, cols, ch = image.shape
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        rotated_image = cv2.warpAffine(image, M, (cols, rows))
        img_conv = Image.fromarray(rotated_image)
        return img_conv
    except Exception as e:
        print(f"Error: {e}. Terjadi kesalahan saat melakukan rotasi pada gambar.")

# Translasi
def translation(image, x, y):
    try:
        rows, cols, ch = image.shape
        M = np.float32([[1, 0, x], [0, 1, y]])
        translated_image = cv2.warpAffine(image, M, (cols, rows))
        img_conv = Image.fromarray(translated_image)
        return img_conv
    except Exception as e:
        print(f"Error: {e}. Terjadi kesalahan saat melakukan translasi pada gambar.")


# Closing
def closing(image, kernel_size=(5, 5)):
    try:
        kernel = np.ones(kernel_size, np.uint8)
        closed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        img_conv = Image.fromarray(closed_image)
        return img_conv
    except Exception as e:
        print(f"Error: {e}. Terjadi kesalahan saat melakukan closing pada gambar.")

# Dilasi
def dilation(image, kernel_size=(5, 5)):
    try:
        kernel = np.ones(kernel_size, np.uint8)
        dilated_image = cv2.dilate(image, kernel, iterations=1)
        img_conv = Image.fromarray(dilated_image)
        return img_conv
    except Exception as e:
        print(f"Error: {e}. Terjadi kesalahan saat melakukan dilasi pada gambar.")

# Erosi
def erosion(image, kernel_size=(5, 5)):
    try:
        kernel = np.ones(kernel_size, np.uint8)
        eroded_image = cv2.erode(image, kernel, iterations=1)
        img_conv = Image.fromarray(eroded_image)
        return img_conv
    except Exception as e:
        print(f"Error: {e}. Terjadi kesalahan saat melakukan erosi pada gambar.")

# Opening
def opening(image, kernel_size=(5, 5)):
    try:
        kernel = np.ones(kernel_size, np.uint8)
        opened_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        img_conv = Image.fromarray(opened_image)
        return img_conv
    except Exception as e:
        print(f"Error: {e}. Terjadi kesalahan saat melakukan opening pada gambar.")

# Brightness
def bright(image, brightness_value):
    try:
        img_bright = cv2.convertScaleAbs(image, beta=brightness_value)
        img_conv = Image.fromarray(img_bright)
        return img_conv
    except Exception as e:
        print(f"Error: {e}. Terjadi kesalahan saat melakukan brightness adjustment pada gambar.")

#Sharpen
def sharpen(image):
    try:
        kernel = np.array([[-1, -1, -1], [-1, 9.5, -1], [-1, -1, -1]])
        image_sharpen = cv2.filter2D(image, -1, kernel)
        img_conv = Image.fromarray(image_sharpen)
        return img_conv
    except Exception as e:
        print(f"Error: {e}. Terjadi kesalahan saat melakukan sharp effect pada gambar.")

#Winter
def winter(img):
    increase_lookup_table = LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
    decrease_lookup_table = LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
    blue_channel, green_channel, red_channel = cv2.split(img)
    red_channel = cv2.LUT(red_channel, decrease_lookup_table).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, increase_lookup_table).astype(np.uint8)
    win = cv2.merge((blue_channel, green_channel, red_channel))
    return win