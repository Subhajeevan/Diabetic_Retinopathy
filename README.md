# Diabetic_Retinopathy
# 🩺 Diabetic Retinopathy Detection using Deep Learning  

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)  
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10-orange?logo=tensorflow)  
![Keras](https://img.shields.io/badge/Keras-DeepLearning-red?logo=keras)  
![OpenCV](https://img.shields.io/badge/OpenCV-ImageProcessing-green?logo=opencv)  
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## 📌 Project Overview  

Diabetic Retinopathy (DR) is a diabetes-related eye disease that can cause blindness if not detected early.  
This project leverages **Deep Learning** and **Transfer Learning** using **ResNet50** to classify **retinal fundus images** into **five stages** of diabetic retinopathy:

| Label            | Description                                |
|-----------------|-------------------------------------------|
| **No_DR**      | No signs of diabetic retinopathy           |
| **Mild**       | Early-stage minor retinal damage          |
| **Moderate**   | Retinal abnormalities present             |
| **Severe**     | Significant retinal damage                |
| **Proliferate_DR** | Advanced stage, high risk of blindness |

---

## 🧠 Key Features

✅ Built using **Transfer Learning** with **ResNet50**  
✅ Handles **class imbalance** using computed **class weights**  
✅ Uses **data augmentation** for better generalization  
✅ Includes **visualizations** for dataset and model performance  
✅ Integrated **early stopping** & **learning rate scheduler** for optimal training  
✅ Designed for **scalability** — can integrate EfficientNet or ViT in future  

---

## 📂 Dataset

- **Source** → [Kaggle: Diabetic Retinopathy Dataset](https://www.kaggle.com/)  
- **Image Size** → 224 × 224 pixels  
- **Classes** → `No_DR`, `Mild`, `Moderate`, `Severe`, `Proliferate_DR`  
- **Split** → 80% Training | 20% Validation  
---

## 🛠 Tech Stack

| Component        | Technology Used |
|------------------|------------------|
| **Programming** | Python 3.10 |
| **Deep Learning** | TensorFlow, Keras |
| **Model** | ResNet50 (Transfer Learning) |
| **Image Processing** | OpenCV, Matplotlib |
| **Data Augmentation** | Keras ImageDataGenerator |
| **Optimization** | Adam Optimizer |
| **Callbacks** | EarlyStopping, ReduceLROnPlateau, ModelCheckpoint |

---

## ⚡ Project Workflow

### **1️⃣ Data Preprocessing & Augmentation**
- Resizes all images to **224×224**
- Normalizes pixel values
- Applies augmentations:
  - Rotation → `20°`
  - Width/Height Shifts → `±20%`
  - Zoom → `20%`
  - Horizontal & Vertical Flips → Enabled

### **2️⃣ Handling Class Imbalance**

from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(all_labels),
    y=all_labels
)
3️⃣ Model Architecture
Base Model: Pre-trained ResNet50 on ImageNet
Custom Layers:
GlobalAveragePooling2D
Dense(512, activation="relu")
Output → Dense(5, activation="softmax")

4️⃣ Model Compilation & Training

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=25,
    class_weight=class_weights,
    callbacks=[early_stop, reduce_lr, checkpoint]
)

5️⃣ Model Evaluation
Plots accuracy & loss curves

Generates confusion matrix

Visualizes predictions on sample images


📉 Training Visualization
Training & Validation Accuracy	Training & Validation Loss

(We can generate actual graphs and embed them.)

🚀 Future Enhancements
🔹 Replace ResNet50 with EfficientNet / Vision Transformers (ViT)
🔹 Use Grad-CAM for visualizing model focus regions
🔹 Perform hyperparameter tuning for better results
🔹 Deploy model as a Streamlit / Flask web app

⭐ Acknowledgments
Dataset and inspiration from Kaggle.

📎 How to Run the Project

# 1. Clone the repository
git clone https://github.com/yourusername/diabetic-retinopathy-detection.git

# 2. Navigate to project folder
cd diabetic-retinopathy-detection

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the notebook
jupyter notebook Diabetic_retinol.ipynb






