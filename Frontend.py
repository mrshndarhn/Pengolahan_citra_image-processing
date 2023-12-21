import streamlit as st
from PIL import Image
import cv2
import numpy as np
from Backend import *

# Fungsi utama untuk menjalankan aplikasi
def main():
    # Menampilkan judul aplikasi
    st.header("Selamat datang di PixelPlay!")
    
    # Mengunggah gambar
    image_upload = st.file_uploader('Silahkan upload gambar...', type=['jpg', 'png', 'jpeg'])
    
    # Memeriksa apakah gambar telah diunggah
    if image_upload is None:
        st.warning('Silakan pilih file gambar yang valid.')
    else:
        try:
            # Membuka gambar dan mengonversinya ke array NumPy
            image = Image.open(image_upload)
            image_cv2 = np.array(image)
            image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2BGRA)
            
            # Memilih filter yang akan digunakan
            option = st.selectbox('Pilih filter yang akan digunakan', ('Edge Detection', 'Grayscale', 'Negative Transformation', 'Gaussian Blur', 'Reduce Noise', 'Sharping', 'Emboising', 'High Boost', 'Low Pass', 'Rotasi', 'Translasi', 'Countour', 'Invers', 'Closing', 'Dilasi', 'Erosi', 'Opening'))
            
            if option:
                st.write('Kamu memilih:', option)
                
                # Tambahkan deskripsi filter
                if option == 'Edge Detection':
                    st.write('Filter ini digunakan untuk mendeteksi tepi pada gambar.')
                elif option == 'Grayscale':
                    st.write('Filter ini mengubah gambar ke skala abu-abu.')
                elif option == 'Negative Transformation':
                    st.write('Filter ini menghasilkan transformasi negatif dari gambar.')
                elif option == 'Gaussian Blur':
                    st.write('Filter ini digunakan untuk memberikan efek blur pada gambar.')
                elif option == 'Reduce Noise':
                    st.write('Filter ini digunakan untuk mengurangi noise pada gambar.')
                elif option == 'Sharping':
                    st.write('Filter ini digunakan untuk mempertajam gambar.')
                elif option == 'Emboising':
                    st.write('Filter ini digunakan untuk menambahkan efek embos pada gambar.')
                elif option == 'High Boost':
                    st.write('Filter ini digunakan untuk menambahkan efek kecerahan pada gambar.')
                elif option == 'Low Pass':
                    st.write('Filter ini digunakan untuk menambahkan efek filter low pass pada gambar.')
                elif option == 'Rotasi':
                    st.write('Filter ini digunakan untuk memutar gambar.')
                    # Memungkinkan pengguna memasukkan sudut rotasi
                    rotation_angle = st.slider('Pilih sudut rotasi:', -180, 180, 0)
                elif option == 'Translasi':
                    st.write('Filter ini digunakan untuk mentranslasikan gambar.')
                elif option == 'Countour':
                    st.write('Filter ini digunakan untuk mengekstraksi kontur pada gambar.')
                elif option == 'Invers':
                    st.write('Filter ini menghasilkan invers dari gambar.')
                elif option == 'Closing':
                    st.write('Filter ini digunakan untuk operasi closing pada gambar.')
                elif option == 'Dilasi':
                    st.write('Filter ini digunakan untuk operasi dilasi pada gambar.')
                elif option == 'Erosi':
                    st.write('Filter ini digunakan untuk operasi erosi pada gambar.')
                elif option == 'Opening':
                    st.write('Filter ini digunakan untuk operasi opening pada gambar.')
                
                # Menampilkan gambar asli
                st.header('Input image')
                st.image(image)
                
                # Menerapkan filter yang dipilih
                with st.spinner(f"Applying {option}..."):
                # Periksa apakah opsi adalah 'Rotasi' dan panggil fungsi rotation dengan sudut yang dipilih
                    if option == 'Rotasi':
                        processed_image = apply_filter(option, image_cv2, rotation_angle)
                    else:
                        processed_image = apply_filter(option, image_cv2)
                
                # Menampilkan gambar setelah diterapkan filter
                st.markdown(f'Image after {option}')
                st.image(processed_image)
        except Exception as e:
            st.error(f'Error: {e}. Pastikan file yang diunggah adalah gambar.')

# Fungsi untuk menerapkan filter yang dipilih
def apply_filter(option, image_cv2, rotation_angle=None):
    if option == 'Edge Detection':
        return edge_detection(image_cv2)
    elif option == 'Grayscale':
        return gray_scale(image_cv2)
    elif option == 'Negative Transformation':
        return negative_transformation(image_cv2)
    elif option == 'Gaussian Blur':
        return Gaussian_Blur(image_cv2)
    elif option == 'Reduce Noise':
        return reduce_noise(image_cv2)
    elif option == 'Sharping':
        return sharp_image(image_cv2)
    elif option == 'Emboising':
        return emboss(image_cv2)
    elif option == 'High Boost':
        return high_boost(image_cv2)
    elif option == 'Low Pass':
        return low_pass(image_cv2)
    elif option == 'Rotasi':
        # Periksa apakah sudut rotasi telah diberikan
        if rotation_angle is not None:
            return rotation(image_cv2, rotation_angle)
        else:
            return image_cv2
    elif option == 'Translasi':
        return translation(image_cv2)
    elif option == 'Countour':
        return contour(image_cv2)
    elif option == 'Invers':
        return inverse(image_cv2)
    elif option == 'Closing':
        return closing(image_cv2)
    elif option == 'Dilasi':
        return dilation(image_cv2)
    elif option == 'Erosi':
        return erosion(image_cv2)
    elif option == 'Opening':
        return opening(image_cv2)

# Menjalankan aplikasi jika file dieksekusi langsung
if __name__ == "__main__":
    main()

