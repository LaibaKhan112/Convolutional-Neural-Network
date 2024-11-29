# Convolutional Neural Network (CNN) for MNIST Classification
This repository contains an implementation of a Convolutional Neural Network (CNN) for classifying the MNIST dataset, which consists of handwritten digits. The model is built using Python, Keras, and TensorFlow.

## Overview
In this project, a CNN is trained on the MNIST dataset to classify images of handwritten digits (0-9). The model includes multiple convolutional layers, pooling layers, and dense layers for feature extraction and classification. The accuracy of the model is evaluated using the test data, and sample predictions are displayed for visual inspection.

## Features
- **Image Classification:** Classify handwritten digits from the MNIST dataset
- **CNN Architecture:** Multiple convolutional and pooling layers followed by dense layers for classification.
- **Model Training:** The model is trained using categorical cross-entropy loss and SGD optimizer.
- **Model Evaluation:** Accuracy of the model is evaluated on the test dataset and visual predictions are shown.

## Requirements
To run this code, you'll need the following libraries:

- tensorflow
- keras
- numpy
- pandas
- matplotlib
- scikit-learn
- 
Install them via pip:
```
pip install tensorflow keras numpy pandas matplotlib scikit-learn

```
## Dataset
This implementation uses the **MNIST dataset**, which is a set of 70,000 28x28 grayscale images of handwritten digits (0-9). The dataset is loaded using Keras's built-in method.

## Model Architecture
The CNN model includes:

- **Conv2D** layers for extracting features.
- **MaxPooling2D** layers for downsampling and reducing the dimensionality of the image.
- **Flatten layer** to reshape the features into a 1D vector.
- **Dense layers** for classification.

## CNN Layers
- 1st Conv2D layer: 6 filters, 3x3 kernel
- 2nd Conv2D layer: 32 filters, 2x2 kernel
- 3rd Conv2D layer: 80 filters, 2x2 kernel
- 4th Conv2D layer: 120 filters, 2x2 kernel
- Fully connected layer with 64 neurons

## Output Layer
The output layer uses softmax activation to classify the digits into 10 categories (0-9).

## Code Walkthrough
1. **Data Preprocessing:**

- The MNIST dataset is loaded and reshaped to fit the model's input shape.
- Pixel values are normalized by dividing by 255.
- The target labels are one-hot encoded using Keras's to_categorical method.
  
2. **Model Training:**

- The model is compiled using the SGD optimizer and categorical cross-entropy loss function.
- The model is trained for 15 epochs with a batch size of 64.
  
3. **Evaluation:**

- The accuracy of the model is evaluated on the test set.
- A subset of predictions is displayed with the predicted labels for visual inspection.


## Usage
1. **Clone the Repository**
   ```
   git clone https://github.com/LaibaKhan112/Convolutional-Neural-Network
   cd cd Convolutional-Neural-Network
   ```

2. **Run the Code**
To run the code, simply execute the following:
   ```
   python Convolutional-Neural-Network.py
   ```
3. **Training Output**
   The training process will print out the loss and accuracy for each epoch. It will also display the final accuracy on the test dataset.

4. **Visualizing Predictions**
   A set of 20 test images will be displayed along with the predicted labels. The model’s performance can be visually validated by comparing predicted and actual labels.




