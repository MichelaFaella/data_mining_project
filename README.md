# Rap Italian Songs – Data Mining Project

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/it/e/e2/Stemma_unipi.svg" width="256" height="256" alt="University of Pisa Logo">
</p>

<p align="center">
  A data mining analysis on Italian rap songs.<br/>
  A project for<br/>
  <b>Data Mining</b><br/>
  course of Computer Science – Artificial Intelligence<br/>
  at <b>University of Pisa</b>.<br/>
  Academic Year <b>2025/2026</b>
</p>

---

## Authors

- **Mary Alrayes** – Mat. 700767  
- **Karol Jinet Cardona Bolanos** – Mat. 702023  
- **Michela Faella** – Mat. 694416  

---

## Project Overview

This project presents a complete **data mining pipeline** applied to a large dataset of **Italian rap music**.
The analysis integrates **linguistic features extracted from lyrics**, **audio descriptors**, **popularity indicators**, and **artist demographic information**.

The work strictly follows the official course guidelines and is structured into five main tasks:
1. Data Understanding & Preparation  
2. Clustering Analysis  
3. Predictive Analysis & Explainable AI  
4. Time Series Analysis on Audio Signals  
5. Ethical and Legal Implications  

All analyses are implemented in **Python** and documented through **well-commented Jupyter notebooks** and a final scientific report.

---

## Dataset Description

The project relies on two main datasets:

### Tracks Dataset
Contains **11,166 Italian rap tracks** with 45 original features, including:
- Track and album metadata
- Linguistic statistics extracted from lyrics
- Popularity indicators (pageviews, popularity scores)
- Audio features (tempo, loudness, pitch, spectral descriptors)

### Artists Dataset
Contains **104 artists** with demographic and geographic information:
- Gender, birth date and place
- Nationality, region, province
- Career start and end dates
- Geographic coordinates

The datasets were merged using the artist identifier and extensively cleaned, transformed, and enriched through feature engineering.

---

## Repository Structure

```

.
├── DM_07_TASK1/
│   ├── data_understanding_and_preparation_1.ipynb
│   ├── data_cleaning_and_preparation_2.ipynb
│   ├── data_transformation_and_correlation_3.ipynb
│   └── project_functions.py
│
├── DM_07_TASK2/
│   ├── clustering_k_means.ipynb
│   ├── clustering_k_Medoids.ipynb
│   ├── clustering_bisectional_k_means.ipynb
│   ├── clustering_density.ipynb
│   └── clustering_hierarchical.ipynb
│
├── DM_07_TASK3/
│   ├── decision_tree.ipynb
│   ├── K-NN.ipynb
│   ├── Ensemble Methods/
│   │   ├── Random_Forest.ipynb
│   │   └── XGBoost.ipynb
│   └── best_knn_params.json
│
├── DM_07_TASK4/
│   └── time_series.ipynb
│
├── data/
│   ├── artists.csv
│   ├── tracks.csv
│   ├── merge_dataset_cleaned.csv
│   ├── merge_dataset_transformed.csv
│   └── mp3_fedexfabri.csv
│
├── README.md
└── DM_Report_07.pdf

````

---

## Tasks

### Task 1 – Data Understanding & Preparation
- Data quality assessment (missing values, duplicates, inconsistencies)
- Feature engineering (linguistic, audio, popularity, demographic indicators)
- Outlier detection and management (IQR, Winsorization, LOF)
- Feature transformations and correlation analysis

### Task 2 – Clustering Analysis
- K-Means, K-Medoids, Bisecting K-Means
- Density-based clustering (DBSCAN, HDBSCAN)
- Hierarchical clustering
- Internal validation using Silhouette, Davies–Bouldin, and Calinski–Harabasz indices

### Task 3 – Predictive Analysis & XAI
- Multi-class classification of artists’ “school of origin”
- Models: Decision Tree, K-NN, Random Forest, XGBoost, EBM
- Explainability using SHAP and LIME
- Best performance achieved by XGBoost

### Task 4 – Time Series Analysis
- Audio feature extraction from MP3 files using Librosa
- Time series clustering (Euclidean vs DTW)
- Motif and anomaly detection via Matrix Profile
- Shapelet extraction for authorship attribution (Fedez vs Fabri Fibra)

### Task 5 – Ethical and Legal Implications
- GDPR compliance considerations
- Data minimization and profiling risks
- Bias and transparency in automated decision-making

---

## Technologies Used

- Python 3
- Pandas, NumPy, Scikit-learn
- Librosa, tslearn
- XGBoost
- SHAP, LIME, InterpretML
- Matplotlib, Seaborn, Plotly

---

## Final Notes

* All notebooks are fully commented
* The final report is provided as `DM_Report_07.pdf`
* The project strictly follows the official course requirements

