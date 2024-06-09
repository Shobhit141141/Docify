from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename
from functions.file_to_text import pdf_to_text  # Assuming this function is in functions/file_to_text.py
import joblib

# Load your pre-trained model
model = joblib.load('best_model.pkl')  # Save your model to a file and load it

# Define the custom category order
custom_category_order = {
    'Legal': 0,
    'Medical': 10,
    'Finance': 80,
    'Education': 40,
    'Business': 50,
    'News': 90,
    'Technical': 30,
    'Creative': 20,
    'Scientific': 60,
    'Government': 70,
}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def upload_file():
    return render_template('index.html')

@app.route('/uploader', methods=['POST'])
def uploader():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Convert PDF to text
        text_content = pdf_to_text(filepath)

        # Join the list of strings into a single string
        input_text = ' '.join(text_content)

        # Make prediction
        predicted_label = model.predict([input_text])
        predicted_category = next(key for key, value in custom_category_order.items() if value == predicted_label[0])

        return jsonify({
            'predicted_category': predicted_category,
            'text_content': input_text  # Return the joined text
        })

if __name__ == '__main__':
    app.run(debug=True)