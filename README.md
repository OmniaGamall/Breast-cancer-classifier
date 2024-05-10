# Breast Cancer Diagnosis Classification
This project aims to classify breast cancer diagnoses as either benign (non-cancerous) or malignant (cancerous) based on various features extracted from diagnostic images. Three different classification algorithms are used: K-Nearest Neighbors (KNN), Support Vector Machine (SVM), and Decision Tree.

## Dataset
The dataset used in this project is the Breast Cancer Wisconsin (Diagnostic) dataset from the UCI Machine Learning Repository. It contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass, which describe characteristics of the cell nuclei present in the image. The dataset includes the following attributes:

. Mean radius
. Mean texture
. Mean perimeter
. Mean area
. Mean smoothness
. Mean compactness
. Mean concavity
. Mean concave points
. Mean symmetry
. Mean fractal dimension
. Standard error (SE) of radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension
. Worst (largest) values of the above features

The target variable is the diagnosis (M = malignant, B = benign).

## Running the Code
The script breast_cancer_classification.py performs the following steps:

1. Loads the dataset from the UCI Machine Learning Repository.
2. Preprocesses the data by removing unnecessary columns and converting the diagnosis labels to binary (0 for benign, 1 for malignant).
3. Splits the dataset into training and testing sets.
4. Trains three different classifiers: KNN, SVM, and Decision Tree.
5. Evaluates the performance of each classifier using accuracy score.
   
## Results
The accuracy scores achieved by each classifier on the testing data are as follows:

1. KNN: [0.9298245614035088]
2. SVM: [0.956140350877193]
3. Decision Tree: [0.9473684210526315]
