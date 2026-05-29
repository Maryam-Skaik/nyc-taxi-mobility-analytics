# 🚖 NYC Taxi Trip Analysis

### Unsupervised Machine Learning & Anomaly Detection Project

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge\&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?style=for-the-badge\&logo=scikitlearn)
![KMeans](https://img.shields.io/badge/KMeans-Clustering-success?style=for-the-badge)
![IsolationForest](https://img.shields.io/badge/Isolation%20Forest-Anomaly%20Detection-red?style=for-the-badge)

---

## 🎯 Project Goal

The goal of this project is to explore real-world transportation behavior in New York City using unsupervised machine learning techniques.

Using the NYC Yellow Taxi Trip dataset, this project focuses on:

* Discovering hidden travel behavior patterns using KMeans clustering
* Evaluating clustering quality using Inertia and Silhouette Score
* Engineering behavioral, temporal, and financial features
* Detecting anomalous and suspicious trips using:

  * KMeans distance-based anomaly detection
  * Isolation Forest
* Understanding operational and financial taxi trip behavior
* Identifying high-confidence anomalies detected by multiple models

This project simulates a real-world mobility analytics and anomaly detection workflow using large-scale transportation data.

---

## 🗂️ Dataset

This project uses the NYC Yellow Taxi Trip Records dataset containing real taxi trip transactions from New York City.

The dataset includes:

* Pickup and dropoff timestamps
* Trip distance
* Passenger count
* Fare information
* Tip amounts
* Toll charges
* Payment types
* Airport and congestion fees

After sampling and preprocessing, the analysis was performed on:

* 185,000+ cleaned taxi trips
* 17 structured features
* Multiple engineered behavioral features

The dataset represents a realistic urban transportation environment with highly diverse trip patterns and naturally occurring anomalies.

---

## 🧹 Data Cleaning & Preprocessing

Several preprocessing steps were applied to ensure data quality and model reliability.

### Cleaning Operations

* Removed invalid negative financial values
* Removed logically inconsistent records
* Handled missing values
* Converted datetime columns into proper timestamp format
* Created trip duration feature from pickup/dropoff timestamps

### Final Clean Dataset

* Original rows: 200,000
* Cleaned rows: 185,401
* Removed invalid rows: 14,599

The cleaned dataset preserves realistic trip variability while eliminating impossible transactional values.

---

## ⚙️ Feature Engineering

To improve clustering quality and anomaly detection performance, several behavioral and temporal features were engineered.

### ⏱️ Temporal Features

* `pickup_hour`
* `pickup_day`
* `is_weekend`

### 💰 Financial Features

* `tip_ratio`
* `fare_per_mile`

### 🚗 Mobility Features

* `trip_duration_min`
* `avg_speed_mph`

These features transformed raw transactional records into richer behavioral representations more suitable for unsupervised learning.

---

## 🤖 Machine Learning Workflow

### 📌 KMeans Clustering

KMeans clustering was used to identify hidden trip behavior patterns within the dataset.

#### Steps Performed

* Selected behavioral and financial features
* Applied feature scaling using `StandardScaler`
* Used:

  * Elbow Method
  * Silhouette Score
* Selected the optimal number of clusters
* Trained the final KMeans model

#### Selected Features

* Trip distance
* Trip duration
* Average speed
* Fare amount
* Tip amount
* Tip ratio
* Fare per mile
* Pickup hour
* Passenger count

---

## 📊 Cluster Analysis

The model segmented taxi trips into distinct behavioral groups.

### 🚕 Cluster 0 — Standard Urban Trips

* Short-distance rides
* Normal trip durations
* Moderate fares
* Represents the majority of daily taxi activity

This cluster reflects the core operational and revenue-driving segment.

---

### ✈️ Cluster 1 — Long Distance / High Value Trips

* Longer travel distances
* Higher fares
* Higher trip durations
* Premium revenue-generating rides

These trips likely represent airport transportation or long-route travel behavior.

---

### 🚦 Cluster 2 — Traffic-Heavy / Slow Trips

* Extremely long durations
* Very low average speed
* Operational inefficiency patterns

This cluster may reflect severe congestion conditions or unusual routing behavior.

---

### ⚠️ Cluster 3 — Anomalous Trips

* Extremely unusual fare-per-mile behavior
* Irregular trip patterns
* Very small cluster size

These records likely represent:

* Pricing inconsistencies
* Rare operational cases
* Data anomalies
* Suspicious transactional behavior

---

## 🚨 Anomaly Detection

Two different anomaly detection approaches were applied to identify unusual taxi trip behavior.

---

### 📍 KMeans Distance-Based Anomaly Detection

Anomalies were identified using the distance between each observation and its nearest cluster centroid.

#### Methodology

* Computed distance to nearest centroid
* Selected top 1% farthest observations
* Flagged these records as anomalies

#### Results

* Detected anomalies: 1,832 trips
* Approximately 1% of dataset

This approach successfully identified:

* Abnormal fare-distance relationships
* Unusual pricing behavior
* Rare mobility patterns

---

### 🌲 Isolation Forest

Isolation Forest was used as a second independent anomaly detection model.

#### Why Isolation Forest?

Unlike KMeans, Isolation Forest isolates rare observations through random partitioning, making it highly effective for high-dimensional anomaly detection.

#### Results

* Detected anomalies: 1,832 trips
* Contamination rate: 1%

Detected anomalies included:

* Extremely expensive rides
* Unusual tipping behavior
* Inconsistent trip duration and fare relationships

---

## 🔍 Cross-Model Validation

To increase anomaly detection reliability, results from both models were compared.

#### Common Anomalies Detected by Both Models

* Shared anomalies: 678 trips

These overlapping observations represent high-confidence anomalies because they were independently detected using two fundamentally different algorithms.

This significantly improves confidence in the identified outliers.

---

## 📈 Key Insights

### 🚕 Most Taxi Trips Follow Predictable Urban Patterns

The majority of trips are:

* Short-distance
* Low-to-moderate fare
* Time-efficient urban rides

---

### 💰 Long-Distance Trips Generate Higher Revenue

A smaller subset of rides contributes disproportionately higher fare values and revenue generation.

---

### 🚦 Traffic Conditions Strongly Affect Operational Efficiency

Slow-speed clusters reveal congestion-heavy trips with poor time efficiency.

---

### ⚠️ Anomalies Reveal Important Edge Cases

Detected anomalies exposed:

* Pricing inconsistencies
* Unusual tipping behavior
* Rare operational cases
* Potential fraudulent or incorrect transactions

---

### 🤝 Combining Multiple Detection Methods Improves Reliability

KMeans and Isolation Forest identified overlapping anomalies, increasing confidence in suspicious trip detection.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

### Machine Learning Models

* KMeans Clustering
* Isolation Forest

---

# ✅ Conclusion

This project demonstrates how unsupervised machine learning can extract meaningful behavioral insights from large-scale transportation data without labeled targets.

Through clustering and anomaly detection techniques, the analysis successfully identified:

* Distinct urban mobility patterns
* Revenue-driving trip segments
* Traffic inefficiencies
* Suspicious and anomalous ride behavior

The project highlights the practical value of unsupervised learning in:

* Transportation analytics
* Fraud detection
* Operational intelligence
* Urban mobility analysis

Overall, this workflow simulates a realistic end-to-end data science pipeline for behavioral analytics and anomaly detection using real-world taxi trip data.

---

# 👨‍💻 Author

Developed as a hands-on machine learning and mobility analytics project focused on unsupervised learning, behavioral clustering, and anomaly detection using real-world NYC transportation data.
