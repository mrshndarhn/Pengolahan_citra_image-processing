import cv2
import numpy as np

def custom_combined_filter(image):
    # Filter pemulusan dengan GaussianBlur
    smoothed_image = cv2.GaussianBlur(image, (5, 5), 0)

    # Filter deteksi tepi dengan Canny
    edges = cv2.Canny(smoothed_image, 50, 150)

    # Menggabungkan citra hasil pemulusan dan deteksi tepi
    combined_image = cv2.addWeighted(smoothed_image, 0.5, edges, 0.5, 0)

    return combined_image

# Membaca citra
image_path = '/image.jpg'
original_image = cv2.imread(image_path)

# Memanggil fungsi custom_combined_filter
result_image = custom_combined_filter(original_image)

# Menampilkan citra asli dan hasil filter kombinasi
cv2.imshow('Original Image', original_image)
cv2.imshow('Combined Filter Result', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()