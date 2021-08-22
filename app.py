import os
from utils.functions import get_latent_repr, get_recon_img_np
from flask import Flask, render_template, request, redirect, url_for, abort
from werkzeug.utils import secure_filename
import utils.models as models
import matplotlib.pyplot as plt
from utils.settings import *

# Load model
model = models.load_model()

app = Flask(__name__)

@app.route("/")
def index():

    return render_template('index.html')


@app.route('/', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    uploaded_filename = secure_filename(uploaded_file.filename)
    if uploaded_file.filename != '':						#if filename is not empty
        file_ext = os.path.splitext(uploaded_filename)[1]			#judging the file extension
        if file_ext not in app.config['UPLOAD_EXTENSIONS']:
            abort(400)
        uploaded_filename = f"img{file_ext}"
        uploaded_file_path = f'{IMAGES_PATH}/{uploaded_filename}'
        uploaded_file.save(uploaded_file_path) #if file extension is acceptable then save it to uploads directory
        

        # bring the image through the model and get the reconstructed image, 
        # and the latent representation of the image (tensor & np)
        latent, latent_np = get_latent_repr(uploaded_file_path, model)
        recon_img_np = get_recon_img_np(latent, model)
        recon_filename = 'recon_img.png'
        print(recon_img_np)
        plt.imsave(f'{IMAGES_PATH}/{recon_filename}', recon_img_np, cmap='gray')
        

        # show the slider, with original values being the latent representation
        # the slider goes from -5 to 5 by default. If the abs() of any of the values of the latent
        # representation are above 5, make the slider range be between those values. 
        # make the step be 100 values

        # add a button with "save compressed_data" to save the latent values into a format
        # that can be decoded later on

    #return redirect(url_for('index'))
    return render_template('uploaded_img.html', img_name = uploaded_filename, recon_img_name = recon_filename)

#new features :
# A playground for changing the latent variables with a slider


if __name__ == '__main__':
	app.run(debug=True)