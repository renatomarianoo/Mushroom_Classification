# Mushroom Image Classification

## Welcome to the Mushroom Image Classification Project!

### Overview

This project is the first endeavor in the Deep Learning Module, aiming to develop a classification model for predicting the common genuses of mushrooms. The project harnesses the capabilities of data science and convolutional neural networks (CNNs) to achieve accurate classifications.

### Background

In the United States, mushroom poisoning accounts for approximately 7,500 cases reported annually. The primary cause is the misidentification of edible mushroom species, a preventable issue through education.

### Project Context

The objective of this project is to build a model capable of classifying mushrooms into their correct genus. Key elements include leveraging transfer learning, exploring various deep learning setups, and experimenting with different network configurations. The dataset is available on Kaggle ([Mushrooms Classification Dataset](https://www.kaggle.com/maysee/mushrooms-classification-common-genuss-images)).

### Models

1. **Model 1 - ResNet18**
2. **Model 2 - MobileNet V2**
3. **Model 3 - ResNet18 with Intermediate Layer**

### Conclusions

The project's main findings include achieving a classification accuracy of approximately 70% using ResNet18-based models. Additional insights suggest that introducing an extra layer did not significantly impact results, and MobileNet V2 produced suboptimal classifications. The analysis of personal mushroom photos highlighted the potential benefits of zooming close to a mushroom for improved classification.

### Improvements

- Evaluate dataset split quality and maintain proportional class representation.
- Experiment with other model backbones, including more complex ones like ResNet50/101.
- Implement class weight balancing during model training.
- Explore additional evaluation metrics for model performance.