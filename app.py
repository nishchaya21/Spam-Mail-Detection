from flask import Flask, request, render_template
from PIL import Image
import pytesseract
import pickle

pytesseract.pytesseract.tesseract_cmd ='/etc/secrets/tesseract'

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectors.pkl", "rb"))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['message']
    input_feature = vectorizer.transform([input_text])
    prediction = model.predict(input_feature)
    # print(prediction[0])
    # return str(prediction[0])
    return render_template("index.html", response_text = "The mail is {}".format("not spam."if str(prediction[0])==1 else "spam."))

@app.route('/convert_image', methods=['POST'])
def convert_image():
    image_file = request.files['image']
    image = Image.open(image_file)
    image = image.convert('L')
    text = pytesseract.image_to_string(image)
    input_vectors = vectorizer.transform([text])
    prediction = model.predict(input_vectors)
    # return str(prediction[0])
    return render_template("index.html", response_text = "The mail is {}".format("not spam."if str(prediction[0])==1 else "spam."))

if __name__ == '__main__':
    app.run(debug=True, port=3000)
