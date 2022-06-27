from flask import Flask, render_template , request , jsonify
from PIL import Image
import os , io , sys
import numpy as np 
import cv2
import base64
from yolo import runModel_yolo
from rnn_mnist import run_model_rnn_mnist
from cnn_cifar10 import run_model_cnn_cifar10
from cnn_cifar100 import run_model_cnn_cifar100
from datetime import datetime



app = Flask(__name__)


DOSSIER_images = './static/images/'


@app.route('/detectObject' , methods=['POST'])
def mask_image():
	#print(request.files , file=sys.stderr)
	#print("request.form :", request.form ) 
	type_nn = request.form['type_nn'] ## byte file
	file = request.files['image'].read()
	file_path = DOSSIER_images + datetime.now().strftime("%m%d%Y%H%M%S") + '_original' + '.jpg'
	with open(file_path, 'wb') as f:
			f.write(file)
			f.close()
	if type_nn=='yolo':
		npimg = np.fromstring(file, np.uint8)
		img = cv2.imdecode(npimg,cv2.IMREAD_ANYCOLOR)

		img = runModel_yolo(img)

		img = Image.fromarray(img.astype("uint8"))
		rawBytes = io.BytesIO()
		img.save(rawBytes, "JPEG")
		rawBytes.seek(0)  
		img_base64 = base64.b64encode(rawBytes.read()) 
		return jsonify({'status':str(img_base64)})
	elif type_nn=='cnn_c10' :
		cnn='cnn_cifar10.h5'
		resultat=run_model_cnn_cifar10(file_path,cnn)
		print(resultat)
	elif type_nn=='cnn_c100':
		cnn='cnn_cifar100.h5'
		resultat=run_model_cnn_cifar100(file_path,cnn)
		print(resultat)
	elif type_nn=='rnn_mnist':
		res=run_model_rnn_mnist(file_path)
		resultat=str(res)
		print(resultat)
	return jsonify({'resultat':resultat})



@app.route('/test' , methods=['GET','POST'])
def test():
	print("log: got at test" , file=sys.stderr)
	return jsonify({'status':'succces'})


@app.route('/')
def home():
	return render_template('./home_upload.html')


@app.route('/view')
def liste_upped():
	images=["images/" + img for img in os.listdir(DOSSIER_images) ] # la liste des images dans le dossier
	return render_template('liste_images.html', images=images)	


@app.after_request
def after_request(response):
    print("log: setting cors" , file = sys.stderr)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response


if __name__ == '__main__':
	app.run(debug = True)
