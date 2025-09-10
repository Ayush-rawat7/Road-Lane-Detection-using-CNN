# Road Lane Detection using Convolutional Neural Networks (CNN)

This project implements a lane detection system using a Convolutional Neural Network (CNN), specifically a U-Net-like architecture. The goal is to identify and highlight lane lines in road images.

## Project Workflow

The project follows a structured workflow to achieve lane detection:

1.  **Set Up the Environment:** Install necessary libraries such as TensorFlow/Keras for building and training the CNN model and OpenCV for image processing.
2.  **Prepare the Dataset:** Utilize a labeled dataset containing road images and corresponding binary masks where lane lines are marked. The Tusimple Lane Detection Dataset is a suitable example.
3.  **Preprocess the Data:** Resize images and masks to a consistent size (e.g., 128x128 pixels) and normalize pixel values. The dataset is split into training, validation, and testing sets.
4.  **Build the CNN Model:** Design a CNN architecture for semantic segmentation. A U-Net-like model is employed, consisting of an encoder for downsampling and a decoder for upsampling with skip connections to preserve spatial information.
5.  **Train the Model:** Compile the model with an appropriate loss function (binary cross-entropy for binary segmentation) and an optimizer (Adam). The model is trained on the training data and validated on the validation set.
6.  **Evaluate the Model:** Assess the trained model's performance on unseen test data.
7.  **Use the Model for Inference:** Apply the trained model to new images or video streams to detect and visualize lane lines.

## Technical Details

### Libraries Used

-   **TensorFlow/Keras:** For building, training, and evaluating the CNN model.
-   **OpenCV (`cv2`):** For image loading, preprocessing, resizing, and visualization.
-   **NumPy:** For numerical operations, especially array manipulation.
-   **Matplotlib:** For plotting training history and visualizing results.

### Model Architecture

The project utilizes a U-Net-like architecture, which is well-suited for image segmentation tasks. The architecture consists of:

-   **Encoder:** Downsampling path with convolutional layers and max pooling to extract features at different scales.
-   **Bottleneck:** A set of convolutional layers at the lowest resolution, capturing high-level features.
-   **Decoder:** Upsampling path with transposed convolutional layers and concatenation with corresponding feature maps from the encoder (skip connections) to recover spatial resolution and refine predictions.
-   **Output Layer:** A 1x1 convolutional layer with a sigmoid activation function to produce a binary mask indicating the presence of lane lines.

The model is compiled with the Adam optimizer and binary cross-entropy loss. Accuracy is used as a metric to monitor training progress.

### Data Loading and Preprocessing

The `load_data` function handles loading images and masks from specified directories. It resizes both to a fixed size (128x128), normalizes image pixel values to the range [0, 1], and converts masks to binary format.

### Training

The model is trained using the preprocessed image and mask data. The training history, including loss and validation loss, is plotted to visualize the learning process.

### Model Saving and Loading

The trained model is saved in the Keras format (`.keras`) for later use. It can be loaded back into memory for evaluation or inference.

### Inference

The `predict_lane` function takes an image and the trained model as input. It preprocesses the image, predicts the lane mask using the model, and resizes the predicted mask to the original image dimensions. The lane mask can then be overlaid on the original image to visualize the detection result.

### Real-Time Detection (Commented Out)

The notebook includes commented-out code for real-time lane detection in videos using `cv2.VideoCapture` and `cv2_imshow`. This part can be uncommented and adapted for video processing.

## Usage

To use this project:

1.  Clone the repository.
2.  Install the required libraries (`tensorflow`, `opencv-python-headless`, `numpy`, `matplotlib`).
3.  Prepare your dataset of road images and lane masks, organizing them into separate directories.
4.  Update the `dataset_path1` and `dataset_path2` variables in the notebook to point to your dataset directories.
5.  Run the notebook cells sequentially to load data, build and train the model, and perform inference on new images or videos.
