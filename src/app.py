from flask import Flask, render_template, request, jsonify
import os
from visualize_and_predict import predict_image
import logging

app = Flask(__name__)

# Logging ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Dosya yüklenmedi'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Dosya seçilmedi'}), 400
    
    # Geçici olarak dosyayı kaydet
    temp_path = 'temp_image.jpg'
    file.save(temp_path)
    
    try:
        # Tahmin yap
        result = predict_image(temp_path)
        
        # Geçici dosyayı sil
        os.remove(temp_path)
        
        return jsonify(result)
    except Exception as e:
        logging.error(f"Tahmin hatası: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 