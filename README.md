# Wildfire Detection using Convolutional Neural Networks (CNN)

## Project Overview
This project uses a Convolutional Neural Network (CNN) to classify images into two categories: '**fire**' or '**nofire**'. The goal is to build a model that can accurately detect the presence of a wildfire in an image, which could be used in automated monitoring systems.

---

## Dataset
The model was trained on **The Wildfire Dataset** from Kaggle. This dataset contains images of fire and no-fire scenes, split into training, validation, and test sets.

- **Dataset Link**: [https://www.kaggle.com/datasets/elmadafri/the-wildfire-dataset](https://www.kaggle.com/datasets/elmadafri/the-wildfire-dataset)

---

## Model Architecture
A sequential CNN was built using TensorFlow's Keras API. The architecture is as follows:

1.  **Input Layer**: Accepts images of shape (150, 150, 3).
2.  **Convolutional Block 1**: A `Conv2D` layer with 32 filters followed by a `MaxPooling2D` layer.
3.  **Convolutional Block 2**: A `Conv2D` layer with 64 filters followed by a `MaxPooling2D` layer.
4.  **Convolutional Block 3**: A `Conv2D` layer with 128 filters followed by a `MaxPooling2D` layer.
5.  **Flatten Layer**: To convert the 2D feature maps into a 1D vector.
6.  **Dense Layers**:
    - A `Dense` layer with 512 units and a 'relu' activation function.
    - A `Dropout` layer with a rate of 0.5 to prevent overfitting.
    - An **output `Dense` layer** with 1 unit and a 'sigmoid' activation for binary classification.

The model was compiled using the **Adam optimizer** and **binary cross-entropy** loss function.

---

## Technologies Used
- **Python**
- **TensorFlow / Keras**: For building and training the neural network.
- **NumPy**: For numerical operations.
- **Matplotlib**: For plotting training history and displaying images.
- **KaggleHub**: For downloading the dataset.

---

## Results
After training for 6 epochs, the model achieved the following performance on the test set:
- **Test Accuracy**: **77.34%**

---

## How to Use
1.  **Setup**: Ensure you have Python and the required libraries installed.
    ```bash
    pip install tensorflow numpy matplotlib kagglehub
    ```
2.  **Load the Model**: Load the trained model file (`FFD.keras`).
    ```python
    from tensorflow.keras.models import load_model
    model = load_model('FFD.keras')
    ```
3.  **Predict on a New Image**: Use the `predict_fire` function provided in the notebook to classify a new image.
    ```python
    # Example usage:
    predict_fire('path/to/your/image.jpg')
    ```
