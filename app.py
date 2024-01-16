#app.py
from flask import Flask, flash, request, redirect, url_for, render_template
# import urllib.request
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from scipy import ndimage
 
app = Flask(__name__)
 
UPLOAD_FOLDER = 'static/uploads/'
UPLOAD_PROCESS_FOLDER = 'static/uploads/processimg'

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['UPLOAD_PROCESS_FOLDER'] = UPLOAD_PROCESS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'webp'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def canny(file):
    img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], file))
    img_canny = cv2.Canny(img,100,200)
    cv2.imwrite(os.path.join(app.config['UPLOAD_PROCESS_FOLDER'], "canny.png"), img_canny)

def robert(file):
    img_float = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], file), 0).astype('float64') 
    img_float/=255.0

    roberts_cross_v = np.array( [[1, 0 ], [0,-1 ]] ) 
    roberts_cross_h = np.array( [[ 0, 1 ], [ -1, 0 ]] ) 
    
    vertical = ndimage.convolve( img_float, roberts_cross_v ) 
    horizontal = ndimage.convolve( img_float, roberts_cross_h ) 
    
    img_robert = np.sqrt( np.square(horizontal) + np.square(vertical)) 
    img_robert*=255
    
    cv2.imwrite(os.path.join(app.config['UPLOAD_PROCESS_FOLDER'], "robert.png"), img_robert)

def prewitt(file):
    img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], file))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gaussian = cv2.GaussianBlur(gray,(3,3),0)

    kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
    img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)  
    img_prewitt = img_prewittx + img_prewitty 
    cv2.imwrite(os.path.join(app.config['UPLOAD_PROCESS_FOLDER'], "prewitt.png"), img_prewitt)
 
@app.route('/')
def home():
    return render_template('index.html')
 
@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):

        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        canny(file.filename)
        prewitt(file.filename)
        robert(file.filename)
        #print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
        return render_template('index.html', filename=filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg, webp')
        return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/process/<filename>')
def process_image(filename):
    return redirect(url_for('static', filename='uploads/processimg/' + filename), code=301)
 
if __name__ == "__main__":
    app.run()