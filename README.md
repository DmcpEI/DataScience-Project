# Heart Disease Prediction with Data Science

> A data science project that analyzes the 2020 CDC health survey to predict heart disease and compare statistical, machine learning, ensemble, deep learning, and clustering approaches.

## 📚 Academic Context

| | |
|---|---|
| **Degree** | BSc in Computer Science @ University of Madeira |
| **Course** | Data Science |
| **Year** | 2023/24 |

## 🛠️ Technologies & Concepts

- Python
- pandas, NumPy, SciPy, statsmodels
- matplotlib and seaborn
- scikit-learn
- TensorFlow / Keras
- UMAP and PCA
- `imbalanced-learn` resampling
- `mlxtend` sequential feature selection
- pickle and joblib model serialization
- Data preprocessing, encoding, outlier removal, and dataset versioning
- Exploratory data analysis and hypothesis testing
- Feature engineering and dimensionality reduction
- kNN from scratch, Logistic Regression, Decision Tree, and MLP
- Bagging, AdaBoost, clustering, and deep learning evaluation

## 🏗️ Architecture / Approach

The project is organized around a full end-to-end pipeline in [`projeto/main.py`](./projeto/main.py), supported by exploratory notebooks in [`projeto/notebooks`](./projeto/notebooks). The workflow starts from the raw CDC dataset in [`projeto/data/heart_2020.csv`](./projeto/data/heart_2020.csv), performs categorical encoding, duplicate and outlier removal, hypothesis testing, and feature creation, and then saves intermediate datasets as cleaned and final CSV files.

The modeling stage splits the data into training, validation, and test sets, applies class rebalancing, and compares multiple approaches: a custom kNN implementation, Logistic Regression, Decision Tree, MLP, Bagging, AdaBoost, a TensorFlow neural model, and clustering methods such as Hierarchical Clustering, K-Means, Gaussian Mixture Models, and OPTICS. The best supervised model is then reused for ensemble learning and sequential backward feature selection.

The repository also includes serialized artifacts such as trained models in [`projeto/models`](./projeto/models), plus [`projeto/builder.pkl`](./projeto/builder.pkl) and [`projeto/feature_selector.pkl`](./projeto/feature_selector.pkl), alongside the project reports that document the mid-project and final phases.

## 🚀 How to Run

1. Create and activate a Python virtual environment.
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   pip install imbalanced-learn
   ```
3. Move into the project directory:
   ```bash
   cd projeto
   ```
4. Run the main pipeline:
   ```bash
   python main.py
   ```
5. Optional: open the notebooks in [`projeto/notebooks`](./projeto/notebooks) for the exploratory and modeling workflow in notebook form.

Running the script generates or updates the processed datasets in `projeto/data/`, evaluates the implemented models, produces plots, and writes serialized artifacts such as `builder.pkl` and `feature_selector.pkl`.

## 📝 Notes

- The project is documented across three deliverables: [`MidJourneyReport.pdf`](./MidJourneyReport.pdf), [`Final Report.pdf`](./Final%20Report.pdf), and [`Executive Summary.pdf`](./Executive%20Summary.pdf).
- The dataset used is the 2020 CDC heart disease survey, and the project follows the full data analytics life-cycle from problem formulation to model comparison.
- The repository already includes generated datasets and saved model artifacts, which makes it possible to inspect both the raw workflow and the resulting outputs.
- The codebase combines exploratory analysis, statistical testing, feature engineering, supervised learning, ensemble methods, deep learning, and clustering in a single project.
