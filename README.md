<img width="1912" height="835" alt="image" src="https://github.com/user-attachments/assets/4366fcbb-8e5a-4d19-a2f8-6e693a52943a" /># Automated Data Preprocessing Application

This project is a Streamlit-based interactive tool designed to simplify and automate the data preprocessing workflow for machine learning. It allows users to upload datasets, clean and transform data, visualize patterns, and download both the processed dataset and step-wise PDF reports. The application is suitable for students, researchers, and professionals who need a fast, no-code preprocessing pipeline.

## Features

# 1. Dataset Upload

* Supports CSV and Excel files
* Displays dataset shape, column details, and statistical summary

# 2. Missing Value Handling

* Detects missing values
* Imputation: Mean, Median, Mode
* Displays the count of missing values handled
* Generates a PDF report after processing

# 3. Outlier Detection

* Supports Z-Score and IQR methods
* Displays number of outliers detected
* Option to remove outliers
* Generates a PDF summary

# 4. Categorical Encoding

* Label Encoding
* One-Hot Encoding
* Displays updated dataset shape
* Includes PDF report

# 5. Feature Scaling

* MinMaxScaler
* StandardScaler
* RobustScaler
* Dataset preview after scaling
* Generates PDF documentation

# 6. Visualization Module

* Line plots, bar charts, and histograms
* Customizable window size for large datasets
* Clean, light-themed interface

# 7. Downloadable Outputs

* Download the processed dataset
* Download step-wise preprocessing PDF files



# Installation

```
pip install streamlit pandas numpy scikit-learn matplotlib seaborn reportlab
```

Clone the repository:

```
git clone https://github.com/your-username/your-repo.git
cd your-repo
```



# Running the Application

```
streamlit run app.py
```



# Usage

1. Upload a dataset (CSV or Excel).
2. Review basic dataset information in the Overview section.
3. Proceed with preprocessing:

   * Handle missing values
   * Detect and remove outliers
   * Encode categorical features
   * Apply scaling
4. Generate and download PDF reports after each step.
5. Download the final cleaned dataset.



# Screenshots

<img width="1912" height="835" alt="image" src="https://github.com/user-attachments/assets/78cc586c-1024-4bee-8a77-9d7c67686363" />
<img width="1896" height="927" alt="image" src="https://github.com/user-attachments/assets/3ffe1b9b-a0e1-44e6-b675-5b033eac6b93" />

<img width="1908" height="926" alt="image" src="https://github.com/user-attachments/assets/1ed8e25c-0307-49d5-96cc-7471a486022d" />



# Technologies Used

* Python
* Streamlit
* Pandas, NumPy
* Scikit-learn
* Matplotlib
* ReportLab

# Author
Divyadarshini K

