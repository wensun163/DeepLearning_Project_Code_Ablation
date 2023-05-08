README file for Ablation_Study_Codes

# Deep Learning for Healthcare Final Project


## Citation to the original paper
This project aims at reproducing the paper 'Readmission prediction via deep contextual embedding of clinical concepts' and check if the reproducer can get similar results presented in the original paper:
https://pubmed.ncbi.nlm.nih.gov/29630604/


## Data Download Instruction:
The dataset in this repository can be downloaded directly from the paper:
https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0195024

In the session 'Supporting information', we can download the 'S1_File.txt'

## Link to the original paper's repo:
The code in the repository is heavily based on the code provided by the author:
https://github.com/danicaxiao/CONTENT


## Guidance to run the code

The below instructions illustrate the methods to preprocess, train, test and evaluate the code in the command:

This folder contains codes for running ablation study experiments on a medical dataset. The dataset is split into three sets: training, validation, and testing. The task is to classify the sequence of events for each patient into different medical codes. The codes are available in the form of pickle files which are loaded into memory using the code in the file Ablation_Study_Codes.ipynb.

## Prerequisites
Access to a GPU and Colab Pro to run the codes efficiently.
The dataset files (in pickle format) should be downloaded from the provided Google Drive link and placed in the same directory as the notebook file.
Running the codes
Mount your Google Drive using the following command:

javascript
Copy code
from google.colab import drive
drive.mount('/content/drive')
Open the 6 Ablation_Study_Codes files in Colab and run the cells in order.

The code contains a function prepare_data that processes the datasets and returns the input matrices for the model. It takes four arguments: seqs, labels, vocabsize, and maxlen. seqs and labels are the input and target sequences respectively. vocabsize is the number of unique codes in the dataset, and maxlen is the maximum length of the input sequence. The function pads the input sequences to the same length and returns matrices x, x_mask, y, lengths, and eventLengths. The matrices x and x_mask are the input and mask matrices respectively, y is the target matrix, lengths is the length of each sequence, and eventLengths is the length of each event in the input sequence.

The code trains a Multi-Output Logistic Regression model or other models using GridSearchCV to find the best hyperparameters. It then evaluates the model on the test set and reports the average accuracy, ROC-AUC, and PR-AUC scores.

## Results

Hyperparameter Tuning

|Model                                | Parameters                                                                                        | ROC-AUC        | 
| ------------------------------------| --------------------------------------------------------------------------------------------------| -------------- | 
| Baseline Model (Logistic Regression)| 'estimator__C': 0.1, 'estimator__penalty': 'l2', 'estimator__solver': 'newton-cg'                 | 0.54+/-0.00    | 
| CNN+ Decision Tree                  | 'estimator__max_depth': 100, 'estimator__min_samples_leaf': 1, 'estimator__min_samples_split': 10 | 0.539+/- 0.001 | 
| CNN+ Logistic Regression            | 'estimator__C': 10.0, 'estimator__penalty': 'l2', 'estimator__solver': 'lbfgs'                    | 0.230+/- 0.07  | 
| KNN                                 | Best parameters: {'estimator__n_neighbors': 7, 'estimator__weights': 'uniform'}                   | 0.538+/- 0.00  | 


## Test Dataset

Experiment results for classification matrics 

| Experiment Run                       | PR-AUC | ROC-AUC | ACC   | CPU/GPU Hours | Memory Usage (GB) |
| -------------------------------------| -------| --------| ------| --------------| ----------------- |
| Baseline Model (Logistic Regression) | 0.282  | 0.492   | 0.798 | 60            |   2500            |
| CNN+ Decision Tree                   | 0.219  | 0.748   | 0.919 | 20            |   4000            |
| CNN+ Logistic Regression             | 0.264  | 0.779   | 0.919 | 20            |   4000            |
| KNN                                  | 0.278  | 0.505   | 0.778 | 50            |   3000            |

## Libraries:

The dependencies required for this project can be found from the file 'dependencyConfirmation.txt':

| Package                 |  Version   | 
| ------------------------| ---------- | 
| numpy                   | 1.20.3     |  
| pandas                  | 1.4.2      |  
| pip                     | 20.0.2     |  
| scikit-learn            | 1.0.2      | 
| google                  | 2.0.3      |
| google-api-core         | 2.11.0     |
| google-api-python-client| 2.84.0     |
| tensorflow              | 2.12.0     |
| cloudpickle             | 2.2.1      |

### Note: All libraries are pre-installed in Colab . No need to redo so.


## Authors
The original dataset and codes were developed by the authors of the paper "Ablation Study of Medical Code Prediction using Deep Learning" (https://arxiv.org/pdf/1906.06627.pdf).

The codes were modified and prepared for this repository by me.