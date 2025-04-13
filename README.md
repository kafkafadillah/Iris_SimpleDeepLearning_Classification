# Iris_SimpleDeepLearning_Classification
Iris Dataset Classification using simple deep learning with 3 layers


```markdown
# 🌸 Iris Flower Classification with Neural Network

This project demonstrates a basic implementation of Deep Learning using the **Iris dataset** from Kaggle to classify iris flower species based on their physical features.

## 📦 Dataset

📌 **Source:** [Kaggle - Iris Dataset](https://www.kaggle.com/datasets/uciml/iris)

### About the Dataset

The Iris dataset was introduced in R.A. Fisher’s classic 1936 paper *The Use of Multiple Measurements in Taxonomic Problems*. It’s also a well-known dataset available in the UCI Machine Learning Repository.

- Total samples: **150**
- Classes: **3 iris species**
  - *Iris-setosa*
  - *Iris-versicolor*
  - *Iris-virginica*
- Each class contains 50 samples

### Features

| Column           | Type    | Description                      |
|------------------|---------|----------------------------------|
| `Id`             | int     | Unique ID for each flower        |
| `SepalLengthCm`  | float   | Sepal length in centimeters      |
| `SepalWidthCm`   | float   | Sepal width in centimeters       |
| `PetalLengthCm`  | float   | Petal length in centimeters      |
| `PetalWidthCm`   | float   | Petal width in centimeters       |
| `Species`        | object  | Flower species (target label)    |

---

## 🧰 Tools & Libraries

- Python
- Pandas, NumPy
- Scikit-learn
- TensorFlow / Keras
- (Optional) Matplotlib for visualization

---

## ⚙️ Project Workflow

1. **Load & Clean Data**
   - Removed the `Id` column since it's not a useful feature
2. **Preprocessing**
   - Normalized numerical features using `StandardScaler`
   - Applied one-hot encoding to the target labels
3. **Train-Test Split**
   - Training set: 80%
   - Test set: 20%
4. **Build Neural Network**
   - 3 Layers:
     - Hidden Layer 1: ReLU
     - Hidden Layer 2: ReLU
     - Output Layer: Softmax
   - Optimizer: `Adam`
   - Loss Function: `CategoricalCrossentropy`
   - Metrics: `Accuracy`
5. **Train for 50 epochs**
6. **Evaluate on Test Data**

---

## 🧠 Model Architecture

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(16, activation='relu', input_shape=(4,)),  # First hidden layer
    Dense(12, activation='relu'),                    # Second hidden layer
    Dense(3, activation='softmax')                   # Output layer
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

---

## 📊 Evaluation Results

| Metric        | Training Set | Test Set     |
|---------------|--------------|--------------|
| Accuracy      | 96.04%       | 92.05%       |
| Loss          | 0.2310       | 0.2679       |

✅ **The model shows excellent performance**, with no major signs of overfitting. High accuracy suggests the neural network successfully learned the patterns in the feature set.

---

## 📈 Suggestions for Improvement

- Add a confusion matrix for deeper evaluation
- Include dropout layers for regularization
- Hyperparameter tuning (e.g. neurons, learning rate, epochs)
- Compare with other models: SVM, KNN, Decision Tree
- Use cross-validation for more robust performance evaluation
