import streamlit as st
from PIL import Image
import cv2
import numpy as np
from Backend import *

# Fungsi utama untuk menjalankan aplikasi
def main():
    # Menampilkan judul aplikasi
    st.header("Selamat datang di PixelPlay!")
    
    st.subheader("Tempat bermain untuk melakukan filter pada gambar")

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
            option = st.selectbox('Pilih filter yang akan digunakan', ('Edge Detection', 'Grayscale', 'Negative Transformation', 'Gaussian Blur', 'Reduce Noise', 'Sharping', 'Emboising', 'High Boost', 'Low Pass', 'Rotasi', 'Translasi', 'Closing', 'Dilasi', 'Erosi', 'Opening', 'Pencil Sketch'))
            
            if option:
                st.write('Kamu memilih:', option)
                
                # Tambahkan deskripsi filter
                if option == 'Edge Detection':
                    st.write('Filter ini adalah teknik untuk menemukan perubahan intensitas yang tajam dalam gambar. Filter ini meninggalkan hanya tepi gambar, menghasilkan gambar yang menonjolkan detail struktural dan kontur.')
                elif option == 'Grayscale':
                    st.write('Filter ini mengubah gambar berwarna menjadi gambar yang hanya memiliki tingkat keabuan. Ini dilakukan dengan menggabungkan informasi warna merah, hijau, dan biru pada setiap piksel dan menghasilkan gambar monokromatik.')
                elif option == 'Negative Transformation':
                    st.write('Filter ini mengubah nilai intensitas warna pada setiap piksel menjadi nilai negatifnya. Ini menghasilkan gambar negatif di mana warna yang semula terang menjadi gelap, dan sebaliknya.')
                elif option == 'Gaussian Blur':
                    st.write('Filter ini meratakan gambar dengan menerapkan filter Gaussian. Ini mengurangi detail tingkat tinggi, menciptakan efek blur, dan dapat digunakan untuk mengurangi noise serta memberikan kesan lebih halus pada gambar.')
                elif option == 'Reduce Noise':
                    st.write('Filter ini mengurangi noise pada gambar tanpa terlalu mempengaruhi detail utama. Metode non-local means denoising digunakan untuk meminimalkan noise yang terlihat sebagai variasi acak dan piksel yang tidak diinginkan.')
                elif option == 'Sharping':
                    st.write('Filter ini digunakan untuk mempertajam gambar dengan menerapkan operasi kernel tertentu. Ini meningkatkan ketajaman dan detail pada gambar.')
                elif option == 'Emboising':
                    st.write(' Filter emboss menambahkan efek tiga dimensi pada gambar dengan menekankan tepi dan perubahan intensitas tinggi. Ini menciptakan ilusi ketajaman dan dimensi.')
                elif option == 'High Boost':
                    st.write(' High Boost filtering meningkatkan ketajaman gambar dengan menggunakan perbedaan antara gambar asli dan versi gambar yang telah dilewati oleh filter low-pass (misalnya, Gaussian Blur).')
                elif option == 'Low Pass':
                    st.write(' Filter low pass meratakan gambar dan menghilangkan komponen frekuensi tinggi. Ini sering digunakan untuk meredam detail halus dan mengurangi noise.')
                elif option == 'Rotasi':
                    st.write('Filter rotasi memutar gambar sebesar sudut tertentu. Ini memungkinkan untuk mengubah orientasi gambar sesuai dengan kebutuhan.')
                    # Memungkinkan pengguna memasukkan sudut rotasi
                    rotation_angle = st.slider('Pilih sudut rotasi:', -180, 180, 0)
                elif option == 'Translasi':
                    st.write('Translasi digunakan untuk memindahkan seluruh gambar ke arah tertentu. Ini membantu dalam menggeser posisi gambar.')
                    # Memungkinkan pengguna memasukkan nilai x dan y untuk translasi
                    translasi_x = st.slider('Masukkan nilai translasi x:', -100, 100, 0)
                    translasi_y = st.slider('Masukkan nilai translasi y:', -100, 100, 0)
                elif option == 'Closing':
                    st.write('Filter ini  melibatkan dilasi diikuti oleh erosi. Ini digunakan untuk menutup celah-celah kecil dalam objek dan menghaluskan tepi.')
                elif option == 'Dilasi':
                    st.write('Filter ini memperluas batas objek dengan menambahkan piksel di sekitarnya. Ini sering digunakan untuk mengisi celah dan memperbesar objek.')
                elif option == 'Erosi':
                    st.write('Filter ini mereduksi ukuran objek dengan menghapus piksel di sekitarnya. Ini berguna untuk menghilangkan detail kecil dan membuat objek lebih kecil.')
                elif option == 'Opening':
                    st.write('Filter ini adalah kebalikan dari closing, yang melibatkan erosi diikuti oleh dilasi. Ini membantu menghilangkan objek kecil dan meratakan tepi.')
                elif option == 'Pencil Sketch':  # Added 'Colour Pencil Sketch Effect'
                    st.write('Filter ini menghasilkan efek sketsa pensil berwarna pada gambar.')
                
                # Menampilkan gambar asli
                st.header('Input image')
                st.image(image)
                
                # Menerapkan filter yang dipilih
                with st.spinner(f"Applying {option}..."):
                # Periksa apakah opsi adalah 'Rotasi' dan panggil fungsi rotation dengan sudut yang dipilih
                    if option == 'Rotasi':
                        processed_image = apply_filter(option, image_cv2, rotation_angle)
                    elif option == 'Translasi':
                        processed_image = apply_filter(option, image_cv2, translasi_x, translasi_y)
                    elif option == 'Pencil Sketch':
                        processed_image = apply_filter(option, image_cv2)
                    else:
                        processed_image = apply_filter(option, image_cv2)

                
                # Menampilkan gambar setelah diterapkan filter
                st.markdown(f'Image after {option}')
                st.image(processed_image)
        except Exception as e:
            st.error(f'Error: {e}.')

# Fungsi untuk menerapkan filter yang dipilih
def apply_filter(option, image_cv2, rotation_angle=None, translasi_x=None, translasi_y=None):
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
        # Periksa apakah nilai translasi x dan y telah diberikan
        if translasi_x is not None and translasi_y is not None:
            return translation(image_cv2, translasi_x, translasi_y)
        else:
            return image_cv2
    elif option == 'Closing':
        return closing(image_cv2)
    elif option == 'Dilasi':
        return dilation(image_cv2)
    elif option == 'Erosi':
        return erosion(image_cv2)
    elif option == 'Opening':
        return opening(image_cv2)
    elif option == 'Pencil Sketch':
        return pencil_sketch_col(image_cv2)
    

# Menjalankan aplikasi jika file dieksekusi langsung
if __name__ == "__main__":
    main()

