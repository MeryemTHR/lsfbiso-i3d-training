# LSFB-ISO Sign Language Recognition (I3D Model)

This project trains an I3D (Inflated 3D Convolutional Neural Network) model on the LSFB-ISO dataset to perform isolated sign language recognition.

## Project Structure
- `datasets/` → Custom Dataset classes for LSFB-ISO
- `models/` → I3D model architecture
- `transforms/` → Preprocessing and video transforms
- `utils/` → Helper functions (dataset loading, training, evaluation)
- `lsfb_i3d_training.ipynb` → Google Colab notebook for training the model

## Dataset

The LSFB-ISO (Belgian French Sign Language - Isolated Signs) dataset contains videos of isolated sign language gestures. The dataset should have the following structure:

```
lsfb_dataset/
├── instances.csv         # Contains sign information (id, sign name)
├── videos/               # Contains MP4 video files
│   ├── vid1.mp4
│   ├── vid2.mp4
│   └── ...
└── metadata/
    └── splits/
        ├── train.json    # List of video IDs for training
        └── test.json     # List of video IDs for testing
```

## Setup and Training

1. Open the `lsfb_i3d_training.ipynb` in Google Colab Pro
2. Upload your LSFB dataset to Google Drive or use the provided dataset path
3. Adjust the `LSFB_DATA_PATH` variable to point to your dataset
4. Run all cells in the notebook to train the model

## Model

The I3D model is initialized with pre-trained weights from the Kinetics dataset and fine-tuned on the LSFB-ISO dataset. The model architecture is based on the Inception-v1 network but inflated to 3D to process video data.

## Training Parameters

- Input: RGB videos with frames of size 224×224, padded to 64 frames
- Batch size: 8
- Learning rate: 1e-4
- Optimizer: Adam with weight decay 1e-5
- Scheduler: OneCycleLR
- Gradient accumulation: 4 steps

## Results

The I3D model achieves significantly better performance on the LSFB-ISO dataset compared to other models like C3D and CNN+RNN approaches, with a test accuracy of around 51%.

## Acknowledgments

This project is based on the [Gesture-Recognition-Experiments](https://github.com/Jefidev/Gesture-Recognition-Experiments) repository, with specific focus on the I3D model for sign language recognition.

The I3D implementation is inspired by the PyTorch implementation from [piergiaj/pytorch-i3d](https://github.com/piergiaj/pytorch-i3d).
