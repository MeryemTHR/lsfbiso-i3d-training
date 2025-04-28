# LSFB-ISO Sign Language Recognition (I3D Model)

This project trains an I3D (Inflated 3D Convolutional Neural Network) model on the LSFB-ISO dataset to perform isolated sign language recognition.

## Project Structure
- `datasets/` ➔ Custom Dataset classes for LSFB-ISO
- `models/` ➔ I3D model architecture
- `transforms/` ➔ Preprocessing and video transforms
- `utils/` ➔ Helper functions (dataset loading, training, evaluation)
- `i3d-train.py` ➔ Main training script

## Training
All training was conducted on Google Colab Pro using a GPU A100 runtime.

## Acknowledgments
Original inspiration: Gesture-Recognition-Experiments repository.
Adaptation and training: MeryemTHR.
