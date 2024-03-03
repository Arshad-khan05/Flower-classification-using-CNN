# Flower Classification

This repository contains code for a flower classification model built using TensorFlow and Keras. The model is capable of classifying images of flowers into one of five categories: Daisy, Dandelion, Rose, Sunflower, and Tulip.

## Requirements

- Python 3.x
- TensorFlow
- Keras
- numpy
- pandas
- OpenCV (cv2)
- Matplotlib
- PIL (Python Imaging Library)

## Getting Started

1. Clone the repository:

   ```bash
   git clone <repository_url>
   cd flower-classification
   ```

2. Ensure you have the required dependencies installed. You can install them using pip:

   ```bash
   pip install tensorflow keras numpy pandas opencv-python matplotlib pillow
   ```

3. Download the dataset and place it in the `/content/drive/MyDrive/flowers` directory.

## Usage

1. Run the `flower_classification.py` script to train the model:

   ```bash
   python flower_classification.py
   ```

2. The model will be trained using the provided dataset, and the trained model will be saved as `OutputFlower.h5` in the specified directory.

3. To classify new flower images, use the provided code snippets in `classify_flowers.py`. You can specify the paths to the images you want to classify.

### Note for Users:

- **Add Your Own Image:** To classify your own flower image, replace the path in the provided code snippets with your desired image path.

- **Use Google Colab:** We recommend using Google Colab for running the code, as it provides free access to GPUs, which can significantly speed up the training process.

You can access the Colab notebook directly by clicking [here](<link_to_your_colab_notebook>).

## Model Architecture

The flower classification model is built using a Convolutional Neural Network (CNN) architecture:

- Input Layer: 224x224x3 pixels
- Convolutional Layers: 4 layers with 64 filters each, followed by max-pooling layers.
- Flatten Layer
- Dense Layers: 1 hidden layer with 512 units, followed by an output layer with 5 units using softmax activation.

## Evaluation

The model is evaluated using the categorical cross-entropy loss function and accuracy metrics. It is trained for 50 epochs with a batch size of 64.

## Sample Classification

You can use the provided code snippets to classify sample flower images. Ensure the paths to the images are correctly specified.

## OutputFlower.h5

The trained model file `OutputFlower.h5` is available in the repository. You can load this model to perform flower classification tasks.

## Author

This flower classification model is developed by [Your Name].

If you have any questions or suggestions, feel free to contact [Your Email].