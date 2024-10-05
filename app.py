from flask import Flask, request, render_template, redirect, url_for
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np

# Inisialisasi Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'  # Ubah folder upload

# Load model
model = load_model('cat_dog_classifier_vgg16.h5')

# Fungsi untuk mempersiapkan gambar
def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Tambahkan dimensi batch
    img_array /= 255.  # Rescale nilai pixel
    return img_array

# Halaman Utama
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Endpoint untuk prediksi
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        # Simpan file yang diupload
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Persiapkan gambar dan prediksi
        img_array = prepare_image(file_path)
        prediction = model.predict(img_array)

        # Tentukan label
        label = 'Dog' if prediction[0][0] > 0.5 else 'Cat'

        # Dapatkan path gambar untuk ditampilkan di template
        img_path = url_for('static', filename='uploads/' + file.filename)

        # Render template dengan hasil prediksi
        return render_template('index.html', prediction=f'Prediction: {label}', img_path=img_path)

if __name__ == '__main__':
    # Buat folder static/uploads jika belum ada
    if not os.path.exists('static/uploads'):
        os.makedirs('static/uploads')
    
    # Jalankan aplikasi
    app.run(debug=True)
