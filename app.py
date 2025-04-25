from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the pre-trained model
try:
    model = load_model('Improved_Skin_Diseases_Diagnosis.h5')
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading model:", str(e))
    exit()

input_shape = model.input_shape[1:3]  # Extract required input shape
print("Model expects input shape:", input_shape)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        try:
            # Preprocess the image
            img = image.load_img(file_path, target_size=input_shape)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0  # Normalize for the model

            # Make prediction
            prediction = model.predict(img_array)
            print("Raw model output:", prediction)  # Log raw predictions
            
            predicted_class = np.argmax(prediction, axis=1)[0]
            confidence = np.max(prediction) * 100

            # Log the predicted class and confidence in the terminal
            print(f"Predicted class index: {predicted_class}")
            print(f"Confidence: {confidence:.2f}%")

            # Disease mapping
            diseases = {
                0: {'name': 'Psoriasis', 'description': 'An autoimmune skin condition causing itchiness and discomfort.'},
                1: {'name': 'Acne', 'description': 'A common skin condition causing pimples due to clogged pores.'},
                2: {'name': 'Eczema', 'description': 'Causes dry, itchy patches of skin. It is not contagious.'},
                3: {'name': 'Warts', 'description': 'Skin growths caused by strains of the human papillomavirus (HPV).'},
                4: {'name': 'Ringworm', 'description': 'A contagious fungal infection causing a ring-shaped rash.'},
                5: {'name': 'Normal skin', 'description': 'Your skin is healthy.'},
            }
            disease_info = diseases.get(predicted_class, {'name': 'Unknown', 'description': 'No description available.'})

            # Log mapped disease info
            print(f"Mapped Prediction: {disease_info['name']} - {disease_info['description']}")

            os.remove(file_path)  # Clean up

            return jsonify({
                'prediction': disease_info['name'],
                'description': disease_info['description'],
                'confidence': f"{confidence:.2f}%"
            })
        except Exception as e:
            print("Error during prediction:", str(e))
            return jsonify({'error': 'Error during prediction'}), 500

    return jsonify({'error': 'Invalid file'}), 400


if __name__ == '__main__':
    app.run(debug=True)