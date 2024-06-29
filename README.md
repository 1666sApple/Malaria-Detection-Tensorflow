# Malaria Detection with TensorFlow
This repository contains code for a Malaria detection project using TensorFlow. The project includes data preprocessing, model creation (ResNet), training, evaluation, and deployment scripts.

## Installation
Clone the repository:
```
git clone https://github.com/1666sApple/Malaria-Detection-Tensorflow.git
cd Malaria-Detection-Tensorflow 
```

## Install Dependencies:
```
pip install -r requirements.txt
```
Make sure you have Python 3.6 or later installed.

## Project Structure

```
.
├── data/
│   ├── data_loader.py
│   └── __init__.py
├── models/
│   ├── resnet_model.py
│   └── __init__.py
├── utils/
│   ├── image_processing.py
│   ├── train_utils.py
│   └── __init__.py
├── main.py
├── train.py
├── evaluate.py
├── .gitignore
├── requirements.txt
└── README.md

```
#### Folder details:
```
`data/`: Directory containing data loading and preprocessing scripts.
`models/`: Directory containing the ResNet model architecture script.
`utils/`: Directory containing utility scripts for image processing and training utilities.
`main.py`: Main script for running the entire pipeline.
`train.py`: Script for training the model.
`evaluate.py`: Script for evaluating the trained model.
`.gitignore`: Ignores specific files.
`requirements.txt`: List of dependencies for the project.
```

## Training
To train the model, run:
```
python train.py
```

This script will train the model using the default parameters and save the trained model in the saved_model/ directory.

## Evaluation

To evaluate the trained model, run:
```
python evaluate.py
```
This script will load the trained model and evaluate its performance on the validation and test datasets.

<!-- ## Contributors
Add your name here if you contributed to this project. -->
<!--  -->
<!-- License -->
<!-- Specify your license (e.g., MIT, Apache 2.0). -->
<!--  -->
## Citation
If you use this code in your research or work, please cite:
```
@article{rajaraman2018pre,
  title={Pre-trained convolutional neural networks as feature extractors toward improved malaria parasite detection in thin blood smear images},
  author={Rajaraman, Sivaramakrishnan and Antani, Sameer K and Poostchi, Mahdieh and Silamut, Kamolrat and Hossain, Md A and Maude, Richard J and Jaeger, Stefan and Thoma, George R},
  journal={PeerJ},
  volume={6},
  pages={e4568},
  year={2018},
  publisher={PeerJ Inc.}
}
```
## Issues
If you encounter any issues or have suggestions, please open an issue.