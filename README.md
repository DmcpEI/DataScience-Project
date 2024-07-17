# Data-Science-Project

## Overview

In this project, we embark on a comprehensive exploration of machine learning techniques to extract actionable insights from the 2020 CDC survey dataset, which includes various metrics and habits of individuals that can influence heart disease, the primary outcome we aim to predict.
The project traverses through the complete data analytics life-cycle, encompassing Problem Formulation, Data Analysis and Cleansing, Model Selection, and Evaluation.

## Table of Contents

- [Introduction](#introduction)
- [Phases of the Project](#phases-of-the-project)
  - [Problem Formulation](#problem-formulation)
  - [Data Analysis and Cleansing](#data-analysis-and-cleansing)
  - [Model Selection](#model-selection)
- [Algorithms and Techniques](#algorithms-and-techniques)
  - [k-Nearest Neighbors (kNN)](#k-nearest-neighbors-knn)
  - [Supervised Learning](#supervised-learning)
  - [Ensemble Models](#ensemble-models)
  - [Deep Learning](#deep-learning)
  - [Feature Selection](#feature-selection)
  - [Clustering Algorithms](#clustering-algorithms)
- [Implementation Details](#implementation-details)
- [Evaluation and Comparison](#evaluation-and-comparison)
- [Conclusion](#conclusion)
- [Future Work](#future-work)

## Introduction

This project documents our journey in implementing and evaluating various machine learning algorithms to uncover patterns and insights from the 2020 CDC survey dataset. Our goal is to demonstrate the transformative potential of machine learning in data analytics.

## Phases of the Project

### Problem Formulation

In the Problem Formulation phase, we articulate a clear definition of the problem addressed by the dataset, outlining the goals and objectives of the ensuing data analysis.

### Data Analysis and Cleansing

In this phase, we delve into pre-processing tasks, describing the dataset’s origins and any preparatory steps undertaken. The report elucidates data cleansing and normalization/standardization processes. Moreover, we navigate through Exploratory Data Analysis (EDA), elucidating descriptive statistics and visualizations employed to comprehend the data. Techniques used include:

- **Descriptive Statistics**: Histograms and correlation analysis.
- **Dimension Reduction**: Both linear (e.g., PCA) and non-linear (e.g., UMAP) methods.

Initial insights gleaned from EDA are discussed, and hypotheses are formulated for further testing.

### Model Selection

This phase entails feature engineering, generating a minimum of 10 new features, initiating model selection, and evaluating suitable model validation methods, all meticulously justified.

## Algorithms and Techniques

### k-Nearest Neighbors (kNN)

- **Implementation**: Crafted from scratch using NumPy arrays.
- **Purpose**: Serve as a foundational model for understanding basic machine learning concepts.
- **Evaluation**: Rigorous performance evaluation on the dataset.

### Supervised Learning

- **Library**: Implemented using the sklearn library.
- **Models**: Tested various models to predict outcomes effectively.
- **Evaluation**: Scrutinized models for their efficacy in predictive tasks.

### Ensemble Models

- **Techniques**: Incorporated bagging and boosting techniques.
- **Purpose**: Enhance predictive accuracy of the models.

### Deep Learning

- **Architecture**: Convolutional Neural Network (CNN) implemented using TensorFlow.
- **Layers**: Comprised multiple layers, including convolutional and pooling layers.
- **Evaluation**: Trained and evaluated on the dataset, compared with ensemble models for optimal performance.

### Feature Selection

- **Model**: Utilized Decision Tree model for feature selection.
- **Purpose**: Enhance understanding of relevant features and optimize model performance.
- **Approach**: Iterative approach to reinforce the importance of feature engineering.

### Clustering Algorithms

- **Techniques**: Applied K-Means, Gaussian Mixture Model (GMM), and Hierarchical Clustering.
- **Purpose**: Uncover inherent data structures and discern patterns and relationships.
- **Evaluation**: Adjusted the number of clusters to gain insights into data distribution and potential groupings.

## Implementation Details

The repository contains detailed documentation of the implementation process for each algorithm and technique used. Each section includes:

- Code files and notebooks
- Documentation and comments within the code
- Performance evaluation metrics

## Evaluation and Comparison

Through meticulous evaluation and comparison, we delineate the strengths and weaknesses of the implemented models. This analysis offers insights into their practical applicability and guides future endeavors in data analytics.

## Conclusion

Our project demonstrates the transformative potential of machine learning in extracting actionable insights from the 2020 CDC survey dataset. The findings pave the way for future work in this domain.

## Future Work

Future work may involve:

- Exploring additional machine learning algorithms
- Further refining feature engineering techniques
- Applying the models to different datasets
- Enhancing model interpretability
