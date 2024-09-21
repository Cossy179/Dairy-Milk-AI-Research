# Milk Quality Prediction Project

This project involves building a machine learning model to predict the grade of milk based on various features such as pH, temperature, fat content, odor, and more. The project uses techniques like classification, feature engineering, clustering, anomaly detection, and regression to provide insights into milk quality and enable predictive capabilities.

## Table of Contents
- [Dataset](#dataset)
- [Project Overview](#project-overview)
- [Techniques and Sections](#techniques-and-sections)
  1. [Exploratory Data Analysis (EDA)](#1-exploratory-data-analysis-eda)
  2. [Clustering Analysis](#2-clustering-analysis)
  3. [Predicting Milk Grade using Classification Model](#3-predicting-milk-grade-using-classification-model)
  4. [Advanced Feature Engineering](#4-advanced-feature-engineering)
  5. [Dimensionality Reduction](#5-dimensionality-reduction)
  6. [Model Evaluation and Optimization](#6-model-evaluation-and-optimization)
  7. [Handling Imbalanced Data](#7-handling-imbalanced-data)
  8. [Regression Analysis](#8-regression-analysis)
  9. [Anomaly Detection](#9-anomaly-detection)
  10. [Conclusion](#10-conclusion)
  11. [Predicting Milk Grade Based on User Input](#11-predicting-milk-grade-based-on-user-input)
- [Requirements](#requirements)
- [How to Run the Project](#how-to-run-the-project)
- [Future Enhancements](#future-enhancements)

## Dataset
The dataset includes the following features:
- **pH**: Acidity level of the milk.
- **Temperature**: Milk temperature in Celsius.
- **Taste**: 0 = bad, 1 = good.
- **Odor**: 0 = bad, 1 = good.
- **Fat**: 0 = absent, 1 = present.
- **Turbidity**: 0 = low, 1 = high.
- **Colour**: Numeric value representing color intensity.
- **Grade**: Quality of milk, categorized as high, medium, or low.

## Project Overview
The goal of this project is to predict the grade of milk based on its features using machine learning. This involves:
- Exploratory data analysis (EDA) to understand the dataset.
- Applying classification models to predict milk grade.
- Advanced techniques like feature engineering, dimensionality reduction, anomaly detection, and handling imbalanced data to improve model performance.

## Techniques and Sections

### 1. Exploratory Data Analysis (EDA)
In the EDA section, we explore the dataset to understand key relationships between features. Basic statistics are calculated, and a correlation matrix is generated to highlight relationships between variables. We also visualize feature distributions to identify trends and outliers. EDA lays the foundation for understanding the dataset before applying machine learning models.

### 2. Clustering Analysis
We use **K-Means Clustering** to group similar data points based on the features of the milk. This helps in identifying natural groupings in the data, such as clusters of high, medium, and low-quality milk. The K-Means algorithm assigns each data point to a cluster, and the results are visualized to see how milk samples are grouped based on their properties.

### 3. Predicting Milk Grade using Classification Model
To predict the grade of milk, we use a **Random Forest Classifier**. The model is trained using a subset of the data and evaluated on the remaining data to ensure it generalizes well to new samples. We also use metrics like accuracy, precision, recall, and F1-score to measure the performance of the model. Feature importance is computed to understand which features contribute most to the prediction.

#### Feature Importance Graph
We also generate a **feature importance graph** that highlights which features are most impactful in predicting the milk grade. This helps us interpret the model and understand the main factors influencing the prediction.

### 4. Advanced Feature Engineering
In this section, we create new features that help improve the predictive power of the model. Examples include:
- **pH to Temperature Ratio**: Capturing the relationship between acidity and heat.
- **Fat-Turbidity Interaction**: Combining fat presence and milk clarity to assess quality.
- **pH Categories**: Converting pH levels into categorical values (Acidic, Neutral, Alkaline).

These new features help the model better capture complex relationships within the dataset.

### 5. Dimensionality Reduction
We apply **Principal Component Analysis (PCA)** to reduce the number of features in the dataset. PCA helps simplify the data by transforming it into a smaller number of principal components that capture the majority of the variance in the original features. We visualize the data in two dimensions to understand patterns more clearly.

### 6. Model Evaluation and Optimization
To ensure the model performs well, we use **k-fold cross-validation** to evaluate its performance. This technique trains the model on different subsets of the data and tests it on the remaining subset, providing a robust evaluation of the model's generalization ability. We also perform **hyperparameter tuning** using **GridSearchCV** to find the best model parameters and optimize performance.

### 7. Handling Imbalanced Data
If the dataset is imbalanced (i.e., some milk grades appear much more frequently than others), the model may struggle to predict the minority classes. To address this, we use **SMOTE** (Synthetic Minority Over-sampling Technique) to generate synthetic data points for the minority class and balance the dataset. This ensures the model performs well across all classes.

### 8. Regression Analysis
In addition to classification, we perform **regression analysis** to predict continuous variables like pH or temperature. Using **Linear Regression**, we model the relationship between features and the continuous target, allowing us to predict pH values based on other milk properties. Mean Squared Error (MSE) is used to evaluate the model’s performance.

### 9. Anomaly Detection
We implement **Anomaly Detection** using the **Isolation Forest** algorithm to identify outliers in the dataset. These outliers may represent unusual or low-quality milk samples. The algorithm assigns anomaly scores to each data point, allowing us to detect samples that differ significantly from the majority of the data.

### 10. Conclusion
In conclusion, we applied a variety of machine learning techniques to the milk dataset to predict its quality and gain deeper insights into the relationships between features. The final model accurately predicts the milk grade and provides valuable information about which features most influence the predictions. This project highlights the importance of data preprocessing, feature engineering, model evaluation, and optimization in building an effective predictive model.

### 11. Predicting Milk Grade Based on User Input
We provide an interactive script that allows users to input data points (e.g., pH, temperature, taste) and predict the milk grade in real-time. The model also provides a confidence score for the prediction, helping users assess how certain the model is about its decision. This interactive tool makes the model practical for real-world applications.

## Requirements
To run the project, you'll need the following Python libraries:

```bash
pip install pandas numpy scikit-learn seaborn matplotlib imbalanced-learn shap
```

## How to Run the Project
1. Clone this repository to your local machine.
2. Install the required Python libraries listed above.
3. Open the Jupyter notebook \`milk_quality_analysis_extended.ipynb\`.
4. Run the notebook to perform all analyses, train the models, and make predictions.
5. Use the interactive script at the end of the notebook to input your own data and get predictions.

## Future Enhancements
- **Deploy the model as a web app**: Build a simple web interface where users can input milk features and receive predictions in real-time.
- **Time-Series Analysis**: If time-based data is available, perform time-series analysis to observe trends in milk quality over time.
- **Advanced models**: Experiment with more complex models like **XGBoost** or **LightGBM** for improved performance.
- **Expand feature engineering**: Add more domain-specific features to enhance model accuracy and interpretability.
