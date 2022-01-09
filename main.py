

from flask import Flask, render_template, request
from keras.models import load_model
from keras import Sequential
from keras.layers import GlobalAveragePooling2D
from sklearn.neighbors import NearestNeighbors
import pickle
import numpy as np
from PIL import Image
import base64
from flask import render_template_string

app = Flask(__name__)

@app.route('/upload')
def display_upload_form():
    return """
    <html>
       <body>
          <pre>{0}</pre>
          <form action = "http://localhost:5000/uploader" method = "POST" 
             enctype = "multipart/form-data">
             <input type = "file" name = "file" />
             <input type = "submit"/>
          </form>
       </body>
    </html>
    """.format(list(feature_list[889]))

@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        img = Image.open(request.files['file'])
        img = img.resize((224, 224)) # resize the image to be 224 x 224
        img = np.array(img) # https://stackoverflow.com/a/47515649
        img = np.expand_dims(img, axis=0) # add a dimension to the np array : (None, 224, 224, 3)

        # 2. ... as the image is being uploaded get feature list
        finetunned_cnn = Sequential()
        finetunned_cnn.add(model_finetuned.get_layer('mobilenet_1.00_224')) # resnet50, mobilenet_1.00_224
        # finetunned_cnn.add(GlobalAveragePooling2D()) # does not work with mobilenet
        feature_list = finetunned_cnn.predict(img)
        # return str(list(feature_list))

        # 3. ... then after you have the feature list, get the most similar feature list
        distances, indices = neighbors.kneighbors(feature_list)
        # return "Distances: " + str(list(distances)) + " <br> " + \
        #         " Indeces: " + str(list(indices))

        return render_template_string('''
        <!DOCTYPE html>
        <html>
            <head>
                <title>Index</title>
            </head>
            <body>
                <img src="data:image/png;base64,{{ img1_str }}"/>
                <img src="data:image/png;base64,{{ img2_str }}"/>
                <img src="data:image/png;base64,{{ img3_str }}"/>
                <img src="data:image/png;base64,{{ img4_str }}"/>
                <img src="data:image/png;base64,{{ img5_str }}"/>
            </body>
        </html>
        ''', img1_str = base64.b64encode(open(filenames[indices[0][0]], 'rb').read()).decode("utf-8"),
            img2_str = base64.b64encode(open(filenames[indices[0][1]], 'rb').read()).decode("utf-8"),
            img3_str = base64.b64encode(open(filenames[indices[0][2]], 'rb').read()).decode("utf-8"),
            img4_str = base64.b64encode(open(filenames[indices[0][3]], 'rb').read()).decode("utf-8"),
            img5_str = base64.b64encode(open(filenames[indices[0][4]], 'rb').read()).decode("utf-8"))

@app.route('/')
def hello_world():
    return 'Hello Flask !!!'

@app.route('/greet')
def greet():
    return '<h1 style="color: red">Hi Mindaugas !!!</h1>'

if __name__ == '__main__':
    # 0. While loading the app, import the weights and the filenames
    feature_list = pickle.load(open('features-caltech101-resnet.pickle', 'rb'))
    filenames = pickle.load(open('filenames-caltech101.pickle', 'rb'))
    model_finetuned = load_model('model-finetuned.h5')

    # 1. ... define the indexer
    neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean').fit(feature_list)
    app.run()