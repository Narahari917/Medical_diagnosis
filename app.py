from flask import Flask,render_template,request,flash,redirect 
import pickle
import numpy as np
from PIL import Image
import tensorflow as tf
import keras
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)


app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

@app.route("/")
def home():
    return render_template('index.html')


@app.route("/services")
def service():
    return render_template('services.html')

@app.route("/symptoms")
def symptoms():
     return render_template('symptoms.html')

@app.route("/faqs")
def faqs():
     return render_template('faqs.html')

@app.route("/about")
def about():
     return render_template('about.html')

@app.route("/asthama")
def asthama():
     return render_template('Diseases/asthama.html')

@app.route("/cancer")
def cancer():
     return render_template('Diseases/cancer_pred.html')

@app.route("/diabetes")
def diabetes():
     return render_template('Diseases/diabetes.html')

@app.route("/heart")
def heart():
     return render_template('Diseases/heart_disease.html')

@app.route("/kidney")
def kidney():
     return render_template('Diseases/kidney_disease.html')

@app.route("/liver")
def liver():
     return render_template('Diseases/liver_disease.html')

@app.route("/obesity")
def obesity():
     return render_template('Diseases/obesity.html')

@app.route("/pneumonia")
def pneumonia():
     return render_template('Diseases/pneumonia.html')

@app.route("/thyroid")
def thyroid():
     return render_template('Diseases/thyroid_reocc.html')


@app.route("/predict_asthama",methods = ['POST','GET'])
def predict_asthama():
     
     try:
          if request.method == 'POST':
               request_dict = request.form.to_dict()
               request_list = list(request_dict.values())
               data_list = []
               for i in request_list:
                    data_list.append(int(i))
               model = pickle.load(open('./models/asthma.pkl','rb'))
               req_arr = np.array(data_list).reshape(1,-1)
               pred = model.predict(req_arr)
               message = ''
               if(pred[0]==1):
                    message = 'The person is having Asthama'
               else:
                    message = 'The person is not having Asthama'
          return render_template('Diseases/asthama.html',pred = message)   
     except:
          message = "please enter valid Data"
          return render_template('index.html',message=message)
     
     

@app.route("/predict_cancer",methods = ['POST','GET'])
def predict_cancer():
     
     try:
          if request.method == 'POST':
               request_dict = request.form.to_dict()
               request_list = list(request_dict.values())
               data_list = []
               for i in request_list:
                    data_list.append((i))
               model = pickle.load(open('./models/cancer.pkl','rb'))
               req_arr = np.array(data_list).reshape(1,-1)
               pred = model.predict(req_arr)
               if(pred[0]==1):
                    message = "The person is having cancer"
               else:
                    message = "The person is not having cancer"
               
          return render_template('Diseases/cancer_pred.html',pred = message)   
     except:
          message = "please enter valid Data"
          return render_template('index.html',message=message)
     
@app.route("/predict_diabetes",methods = ['POST','GET'])
def predict_diabetes():
     
     try:
          if request.method == 'POST':
               request_dict = request.form.to_dict()
               request_list = list(request_dict.values())
               data_list = []
               for i in request_list:
                    data_list.append(float(i))
               model = pickle.load(open('./models/diabetic.pkl','rb'))
               req_arr = np.array(data_list).reshape(1,-1)
               pred = model.predict(req_arr)
               if(pred[0]==1):
                    message = "The person is having diabetes"
               else:
                    message = "The person is not having diabetes"
               
          return render_template('Diseases/diabetes.html',pred = message)   
     except:
          message = "please enter valid Data"
          return render_template('index.html',message=message)
     

@app.route("/predict_heart",methods = ['POST','GET'])
def predict_heart():
     
     try:
          if request.method == 'POST':
               request_dict = request.form.to_dict()
               request_list = list(request_dict.values())
               data_list = []
               for i in request_list:
                    data_list.append(float(i))
               model = pickle.load(open('./models/heart.pkl','rb'))
               req_arr = np.array(data_list).reshape(1,-1)
               pred = model.predict(req_arr)
               if(pred[0]==0):
                    message = "The person is having heart disease"
               else:
                    message = "The person is not having heart disease"
               
          return render_template('Diseases/heart_disease.html',pred = message)   
     except:
          message = "please enter valid Data"
          return render_template('index.html',message=message)
     
@app.route("/predict_kidney",methods = ['POST','GET'])
def predict_kidney():
     
     try:
          if request.method == 'POST':
               request_dict = request.form.to_dict()
               request_list = list(request_dict.values())
               data_list = []
               for i in request_list:
                    data_list.append(float(i))
               model = pickle.load(open('./models/kidney.pkl','rb'))
               req_arr = np.array(data_list).reshape(1,-1)
               pred = model.predict(req_arr)
               if(pred[0]==1):
                    message = "The person is having kidney disease"
               else:
                    message = "The person is not having kidney disease"
               
          return render_template('Diseases/kidney_disease.html',pred = message)   
     except:
          message = "please enter valid Data"
          return render_template('index.html',message=message) 
     

@app.route("/predict_liver",methods = ['POST','GET'])
def predict_liver():
     
     try:
          if request.method == 'POST':
               request_dict = request.form.to_dict()
               request_list = list(request_dict.values())
               data_list = []
               for i in request_list:
                    data_list.append(float(i))
               model = pickle.load(open('./models/liver.pkl','rb'))
               req_arr = np.array(data_list).reshape(1,-1)
               pred = model.predict(req_arr)
               if(pred[0]==1):
                    message = "The person is having liver disease"
               else:
                    message = "The person is not having liver disease"
               
          return render_template('Diseases/liver_disease.html',pred = message)   
     except:
          message = "please enter valid Data"
          return render_template('index.html',message=message) 



@app.route("/predict_obesity",methods = ['POST','GET'])
def predict_obesity():
     
     try:
          if request.method == 'POST':
               request_dict = request.form.to_dict()
               request_list = list(request_dict.values())
               data_list = []
               for i in request_list:
                    data_list.append(float(i))
               model = pickle.load(open('./models/obesity.pkl','rb'))
               req_arr = np.array(data_list).reshape(1,-1)
               pred = model.predict(req_arr)
               if(pred[0]==1):
                    message = "The person is having obesity"
               else:
                    message = "The person is not having obesity"
               
          return render_template('Diseases/obesity.html',pred = message)   
     except:
          message = "please enter valid Data"
          return render_template('index.html',message=message) 

@app.route("/predict_pneumonia", methods=['POST', 'GET'])
def predict_pneumonia():
     message = ''
     if request.method == 'POST':
        try:
            if 'image' in request.files:
                img = Image.open(request.files['image']).convert('L')
                img = img.resize((36,36))
                img = np.asarray(img)
                img = img.reshape((1,36,36,1))
                img = img / 255.0
                model = load_model("models/pneumonia.h5")
                pred = np.argmax(model.predict(img)[0])
                
                if (pred==0):
                    message = "The person is not having pneumonia"
                else:
                    message = "The person is having pneumonia"
        except:
            message = "Please upload an Image"
            return render_template('pneumonia.html', message = message)
     return render_template('./Diseases/pneumonia.html', pred = message)


    

@app.route("/predict_thyroid",methods = ['POST','GET'])
def predict_thyroid():
     
     try:
          if request.method == 'POST':
               request_dict = request.form.to_dict()
               request_list = list(request_dict.values())
               data_list = []
               for i in request_list:
                    data_list.append(float(i))
               model = pickle.load(open('./models/thyroid.pkl','rb'))
               req_arr = np.array(data_list).reshape(1,-1)
               pred = model.predict(req_arr)
               if(pred[0]==1):
                    message = "The person is having a chance to get thyroid again"
               else:
                    message = "The person is not having a chance to get thyroid again"
               
          return render_template('Diseases/thyroid_reocc.html',pred = message)   
     except:
          message = "please enter valid Data"
          return render_template('index.html',message=message) 



if __name__ == '__main__':
	app.run(debug = True)
     