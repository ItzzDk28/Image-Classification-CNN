from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load the trained CNN model
model = load_model('cnn_cifar10_model.keras')

# Define class names
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'}), 400
    
    img_file = request.files['image']
    if img_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Load and preprocess the image
        img = image.load_img(img_file, target_size=(32, 32))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction))

        return jsonify({
            'predicted_class': predicted_class,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)