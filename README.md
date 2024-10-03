# Breast Cancer Diagnosis with Neural Networks

## Project Overview
This project uses a neural network model to classify breast cancer samples as either benign (B) or malignant (M) based on features derived from cell nuclei. The dataset contains 30 real-valued features calculated for each cell nucleus, and the model is trained to predict the diagnosis based on these features. The project implements data preprocessing, feature scaling, model training, and predictions using Python libraries like TensorFlow/Keras, pandas, and scikit-learn.

## 1. Objective
The goal of this project is to build a neural network classifier to predict whether a given sample of breast cancer cells is benign or malignant. The model will be trained on historical data of breast cancer diagnoses, with various features related to cell nucleus properties.

## 2. Dataset
The dataset contains a total of 32 columns:
- **ID**: Identification number for each sample (not used for training).
- **Diagnosis**: The target variable, where `M` indicates a malignant diagnosis, and `B` indicates a benign diagnosis.
- **Features (3-32)**: 30 real-valued features that describe the physical properties of each cell nucleus. These include:
  - Radius (mean of distances from center to points on the perimeter)
  - Texture (standard deviation of gray-scale values)
  - Perimeter
  - Area
  - Smoothness (local variation in radius lengths)
  - Compactness (perimeter² / area - 1.0)
  - Concavity (severity of concave portions of the contour)
  - Concave points (number of concave portions of the contour)
  - Symmetry
  - Fractal dimension (coastline approximation - 1)

## 3. Data Preprocessing
- **Feature Scaling**: Before feeding the data into the neural network, we applied `StandardScaler` to normalize all feature values, ensuring that they are within a common range and preventing bias towards features with larger magnitudes.
- **Train-Test Split**: The data is split into 70% for training and 30% for testing, ensuring that the model is evaluated on unseen data.

## 4. Solution Outline

### 4.1 Data Normalization
Data normalization is essential in neural networks to ensure that all features are on the same scale. The `StandardScaler` from `scikit-learn` was used to normalize the features.

### 4.2 Model
The project uses a simple feedforward neural network built with `TensorFlow`/`Keras`:
- **Input Layer**: 30 features (after normalization).
- **Hidden Layers**: Multiple dense layers with ReLU activation.
- **Output Layer**: A single neuron with sigmoid activation for binary classification (benign or malignant).
- **Loss Function**: Binary cross-entropy.
- **Optimizer**: Adam optimizer for efficient gradient descent.

#### 4.2.1 Basic Concepts
- **Sigmoid Function**: The sigmoid function outputs a probability value between 0 and 1, making it suitable for binary classification tasks.
- **Binary Cross-Entropy**: Measures the difference between the predicted probability and the actual class, guiding the network to improve its predictions.

## 5. Conclusion

### 5.1 Model Results
After training, the model is evaluated on the test set, and metrics such as accuracy, precision, recall, and the confusion matrix are used to measure its performance. The neural network achieves high accuracy in predicting the diagnosis, making it a useful tool for assisting in breast cancer diagnosis.

## 6. Installation and Usage

### Prerequisites
- Python 3.x
- TensorFlow/Keras
- pandas
- scikit-learn
- matplotlib
- seaborn

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repository/breast-cancer-diagnosis.git
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage
1. **Train the Model**: Run the script to load the data, preprocess it, train the model, and evaluate it.
   ```bash
   python cancer.py
   ```

2. **Make Predictions**: To use the trained model on new data, modify the code with your new data samples and make predictions:
   ```python
   # Replace the sample data with real feature values
   new_sample = np.array([[12.34, 14.56, ..., 0.06543]])
   prediction = model.predict(new_sample)
   ```

3. **Visualization**: The script also provides visualizations of the data distribution, correlation matrix, and boxplots for feature analysis.

### Example Prediction
```python
# Example of making a prediction on a new sample
new_sample = np.array([[12.34, 14.56, 85.67, 520.0, 0.09234, ..., 0.06543]])
new_sample_normalized = scaler.transform(new_sample)
prediction = model.predict(new_sample_normalized)
print("Malignant" if prediction > 0.5 else "Benign")
```

## 7. Visualization
- **Correlation Matrix**: Shows the relationships between features, highlighting those that are most correlated.
- **Count Plot**: Visualizes the distribution of benign and malignant samples in the dataset.
- **Boxplots**: Provide insights into how each feature differs between benign and malignant samples.

## 8. References
- Wolberg, W. H., Street, W. N., & Mangasarian, O. L. (1995). "Computerized Breast Cancer Diagnosis."
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
- Scikit-learn documentation: https://scikit-learn.org
- TensorFlow documentation: https://www.tensorflow.org
