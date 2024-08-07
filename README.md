# PIoT-Oriented Multi-Target Recognition of Substation Infrared Images

[![forthebadge](https://forthebadge.com/images/badges/built-with-love.svg)](https://forthebadge.com) [![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg)](https://forthebadge.com)

This repository contains the Python implementation of a deep learning-based multi-target recognition system for substation infrared images, specifically designed for the Power Internet of Things (PIoT). The system is built upon an optimized Faster R-CNN architecture, enhanced with class rectification inspired by Non-Maximum Suppression (NMS), to accurately identify various types of electrical equipment and their parts within substation environments.

## Overview

The proposed method addresses the challenge of accurately recognizing a wide range of electrical equipment and their components in substation infrared images. By leveraging the powerful feature extraction capabilities of ResNet-101 and the efficient object detection framework of Faster R-CNN, the system achieves high recognition accuracy, which is crucial for timely fault detection and preventive maintenance in PIoT.

## Key Features

- **Optimized Faster R-CNN Architecture**: The core of the system, enhanced with mechanisms for improved recognition of substation equipment.
- **Class Rectification**: A novel approach to correct misclassifications between equipment and their parts, inspired by NMS.
- **High Recognition Accuracy**: Outperforms other object detection methods on the infrared dataset, as demonstrated in the comparison study.


## Results

The system achieves state-of-the-art results on the infrared dataset, with significant improvements in recognition accuracy compared to other object detection methods. Detailed results and analysis can be found in the paper.


## Getting Started

To get started with this project, you'll need to have Python installed on your system, along with several Python packages. The following instructions will guide you through setting up the environment, preparing the dataset, and running the code.

### Prerequisites

- Python 3.6 or higher
- PyTorch 1.7.0 or higher
- torchvision 0.8.1 or higher
- OpenCV 3.4.0 or higher
- NumPy 1.18.1 or higher

### Installation

1. Clone this repository:

```bash
git clone https://github.com/limin427/PIoT-Oriented-Multi-Target-Recognition-of-Substation-Infrared-Images-Driven-by-Deep-Learning.git
cd PIoT-Oriented-Multi-Target-Recognition-of-Substation-Infrared-Images-Driven-by-Deep-Learning
```

2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

### Preparing the Dataset

1. Prepare your infrared substation image dataset.
2. Annotate the images with bounding boxes for the objects of interest.
3. Organize your dataset in the following structure:

```
infrared_dataset/
    train/
        images/
        annotations/
    val/
        images/
        annotations/
```


### Running the Code

1. Adjust the configuration in the `main.py` script to match your dataset and desired settings.
2. Run the training script:

```bash
python main.py --data-path /path/to/your/dataset --num-classes 16 --output-dir /path/to/save/outputs
```

3. Monitor the training progress and wait until it completes. The trained result will be saved in the specified output directory.

### Customization

- To customize the model, you can modify the `FasterRCNN` class and its components, such as the ResNet101 backbone, RPN, RoIHead, and RoIAlign layers.
- Adjust the hyperparameters in the `main.py` script or the argument parser to fine-tune the training process.
- Implement additional functions or classes as needed to extend the functionality of the system.


## Contributing

If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request. Contributions are always welcome!
