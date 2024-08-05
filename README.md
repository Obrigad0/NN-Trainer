# Neural Network Training and Testing Script

## Overview

This repository contains two powerful scripts designed for training and testing neural network models. The scripts are versatile and can be easily adapted to various model architectures and datasets.

## Features

- **Flexible Model Architecture**: Currently configured for a Fully Connected Neural Network (FCNN), but easily adaptable to other architectures like Convolutional Neural Networks (CNN).
- **Dataset Versatility**: Pre-configured for MNIST, but can be used with any desired dataset by adjusting input and output settings.
- **Real-time Training Feedback**: Provides updates on training and testing accuracy during the model training process.
- **Visual Prediction Test**: After training, the script showcases a sample image from the dataset along with the model's prediction.
- **Pre-trained Model Testing**: The second script allows testing of pre-trained models on standard or custom datasets, useful for evaluating models with potential backdoors or watermarks.

## Usage

### Training Script

1. Set your desired model architecture in the script (default is FCNN).
2. Configure the input and output settings to match your chosen dataset.
3. Run the script to start training.
4. Monitor the console for real-time accuracy updates.
5. At the end of training, view the sample prediction test.

### Testing Script

1. Load your pre-trained model.
2. Choose between the default dataset (MNIST) or a custom dataset.
3. Run the script to evaluate the model's performance.
4. Analyze the results, particularly useful for backdoor/watermark accuracy testing.

## Customization

The scripts are designed to be highly customizable. You can easily modify:

- Model architecture
- Dataset choice
- Input/output dimensions
- Training parameters (epochs, batch size, etc.)
- Testing procedures
