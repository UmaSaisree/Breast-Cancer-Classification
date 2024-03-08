# Convolutional Neural Network Improvement for Breast Cancer Classification (CNNI-BCC)

## Introduction

Breast cancer is the most common malignancy among women worldwide, affecting millions every year with varying prognosis based on early detection and treatment. The statistical data from 2017 indicates a significant incidence rate, with over 252,710 new invasive cases and 63,410 non-invasive cases reported in the United States alone. This highlights the critical need for early and accurate diagnostic procedures.

Traditional mammography, while being the standard screening method, has its limitations, especially for younger women and those with dense breast tissues. Advanced imaging techniques like CE digital mammography, ultrasound, and MRI have improved diagnostic accuracy but come with drawbacks like high costs, limited availability, and potential over-diagnosis.

The Convolutional Neural Network Improvement for Breast Cancer Classification (CNNI-BCC) project aims to address these challenges by utilizing deep learning to enhance the classification accuracy of mammographic images. The CNNI-BCC model proposes a novel and less invasive approach to assist medical experts in the early detection and classification of breast cancer tissues, potentially reducing the reliance on scarce medical specialists and contributing to a higher survival rate through timely and precise treatment interventions.

This repository contains the implementation details, dataset descriptions, and evaluation results of the CNNI-BCC, which underscores the transformative impact of machine learning in medical diagnostics.

## Methodology

In this project, we adopt a data-driven approach to improve breast cancer classification using mammographic images. Our methodology centers around harnessing the power of convolutional neural networks (CNNs) and innovative data augmentation techniques to enhance the model's accuracy and reliability.

### Dataset Description

[Kaggle dataset page](https://www.kaggle.com/datasets/kmader/mias-mammography)

The MIAS Mammography dataset is leveraged in this project, encompassing mammographic images for the automation of breast cancer detection.

#### Content
The dataset includes mammography scans with labels and annotations. 

#### Labels
The labels consist of various attributes:
- **Reference Number**: Unique identifier within the MIAS database.
- **Background Tissue Character**: 
  - `F` for Fatty
  - `G` for Fatty-glandular
  - `D` for Dense-glandular
- **Abnormality Class**:
  - `CALC` for Calcification
  - `CIRC` for Well-defined/circumscribed masses
  - `SPIC` for Spiculated masses
  - `MISC` for Other, ill-defined masses
  - `ARCH` for Architectural distortion
  - `ASYM` for Asymmetry
  - `NORM` for Normal
- **Severity**:
  - `B` for Benign
  - `M` for Malignant
- **Abnormality Center Coordinates**: Given in x,y pixel format.
- **Radius**: Approximate radius in pixels covering the abnormality.

Images are standardized to 1024x1024 pixels and centered in the matrix. 


### Feature Wise Data Augmentation

FWDA mitigates the challenge of high-dimensional data by generating augmented images through rotations and flips. This technique not only enhances the dataset but also introduces variability, enabling the CNN model to capture a wider array of patterns within the mammographic images, which is critical for improving the model's ability to generalize.

**Algorithm** :

1. Input: Set of raw mammogram images of size 1024x1024 pixels.
2. Output: Augmented image patches of size 128x128 pixels.
3. For each image in the dataset:
   1. Perform rotation transformations at angles of 90°, 180°, and 270°
   2. Apply vertical flipping to each rotated image.
   3. Save each transformed image as a separate patch.
4. Compile all generated patches into an augmented dataset.
5. End

### Convolutional Neural Networks Based Classification

With the pre-processed and augmented dataset, we implement a CNN-based classification system. The CNN model is structured with convolutional layers for feature extraction, pooling layers for dimensionality reduction, and densely connected layers for final classification. This architecture is specifically tailored to identify subtle and distinctive patterns indicative of benign or malignant breast cancer indicators in mammographic images.

Through this methodological framework, our project aims to refine the efficiency and effectiveness of breast cancer detection, offering a promising tool to support medical diagnostics.

**Algorithm** :

1. Input: Augmented image patches from FWDA.
2. Output: Classification into 'benign', 'malignant', or 'normal'.
3. Initialize a CNN with specified architecture parameters.
4. Train the CNN using the augmented dataset:
   1. Apply feedforward computation to derive predictions.
   2. Calculate loss using the categorical cross-entropy function.
   3. Perform backpropagation to update model weights.
5. Validate the CNN model using a separate validation set.
6. End

## Results

Our model demonstrates promising performance in classifying mammographic images into benign, malignant, or normal categories. Key highlights from our evaluation include:

Area Under the Curve (AUC): The model achieved an AUC score of 0.901, indicating a high degree of separability between the classes.

Confusion Matrix: The confusion matrix revealed **Sensitivity** of 89.47 and **Specificity** 90.71, showcasing the model's ability to correctly identify the majority of cases across all categories.


## Conclusion

The CNNI-BCC method, aimed at aiding medical professionals in diagnosing breast cancer, leverages a supervised deep learning neural network to improve the classification process. Traditionally reliant on manual segmentation, this approach is time-consuming and subject to human and machine error. Tested on 221 real patient subjects, CNNI-BCC demonstrated superior performance over existing methods and commercial products, achieving high sensitivity, accuracy, AUC, and specificity.
