  # 🏥 Medical Diagnosis System

A Flask-based web application that predicts the likelihood of **9 diseases** from patient symptoms and medical data using trained machine learning models. Includes an image-based pneumonia detector powered by a Convolutional Neural Network (CNN).

---

## 🔍 Supported Diseases

| Disease | Input Type | Model Type |
|---|---|---|
| Asthma | Symptom data | Classification (sklearn) |
| Cancer | Clinical features | Classification (sklearn) |
| Diabetes | Medical metrics | Classification (sklearn) |
| Heart Disease | Clinical features | Classification (sklearn) |
| Kidney Disease | Lab values | Classification (sklearn) |
| Liver Disease | Lab values | Classification (sklearn) |
| Obesity | Physical metrics | Classification (sklearn) |
| Pneumonia | Chest X-ray image | CNN (Keras/TensorFlow) |
| Thyroid Recurrence | Clinical features | Classification (sklearn) |

---

## 🏗️ Project Structure

```
Medical_diagnosis/
├── app.py                  # Flask application — routes and prediction logic
├── models/                 # Trained model files (.pkl and .h5)
│   ├── asthma.pkl
│   ├── cancer.pkl
│   ├── diabetic.pkl
│   ├── heart.pkl
│   ├── kidney.pkl
│   ├── liver.pkl
│   ├── obesity.pkl
│   ├── pneumonia.h5        # CNN model for image-based prediction
│   └── thyroid.pkl
├── Medical/                # HTML templates for each disease page
│   └── Diseases/
├── notebooks/              # Jupyter notebooks for model training and EDA
└── README.md
```

---

## ⚙️ Tech Stack

- **Backend:** Python, Flask
- **ML / DL:** scikit-learn, TensorFlow, Keras
- **Image Processing:** PIL (Pillow), NumPy
- **Model Serialization:** Pickle (.pkl), Keras HDF5 (.h5)
- **Frontend:** HTML, CSS (Jinja2 templates)

---

## 🚀 How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Narahari917/Medical_diagnosis.git
cd Medical_diagnosis
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> If a `requirements.txt` is not present, install manually:
> ```bash
> pip install flask numpy pillow tensorflow scikit-learn keras
> ```

### 3. Run the app

```bash
python app.py
```

### 4. Open in browser

```
http://127.0.0.1:5000
```

---

## 🧠 How It Works

**For symptom-based diseases (8 models):**
1. User fills out a form with clinical or symptom values
2. Flask collects the form data and passes it to the loaded `.pkl` model
3. The model returns a binary prediction (disease present / not present)
4. Result is displayed back on the same page

**For pneumonia (CNN model):**
1. User uploads a chest X-ray image
2. Image is converted to grayscale, resized to 36×36, and normalized
3. The Keras CNN model predicts the class (pneumonia / normal)
4. Result is displayed on the pneumonia page

---

## 📓 Notebooks

The `notebooks/` folder contains Jupyter notebooks used to:
- Perform exploratory data analysis (EDA) on each dataset
- Train and evaluate classification models
- Export trained models as `.pkl` files for Flask integration

---

## 🔮 Future Improvements

- Add model confidence scores alongside predictions
- Improve frontend UI with a modern framework
- Add input validation and error handling on the frontend
- Deploy to a cloud platform (Render, Railway, or AWS)
- Expand dataset coverage and retrain models for better accuracy

---

## 👨‍💻 Author

**Narahari Kommi**
[LinkedIn](https://www.linkedin.com/in/naraharikommi/) | [GitHub](https://github.com/Narahari917)

---

## 📄 License

This project was built for educational purposes. Add your preferred license here.
