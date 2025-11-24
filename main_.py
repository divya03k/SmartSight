# app.py
# SmartSight - Full App (with ML module)
# Save and run with: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO, StringIO
import os
from datetime import datetime
import json
import hashlib

# optional libs
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except Exception:
    WORDCLOUD_AVAILABLE = False

# sklearn for ML module
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

st.set_page_config(page_title="SmartSight – Universal Data Analyzer", layout="wide")
st.title("📊 SmartSight – Analyze. Clean. Visualize. Automate.")

# ------------------ Helpers ------------------

def is_datetime_series(s, sample_size=50):
    s_nonnull = s.dropna()
    if s_nonnull.empty:
        return False
    try:
        sample = s_nonnull.sample(min(sample_size, len(s_nonnull)), random_state=1).astype(str)
        pd.to_datetime(sample, errors='raise')
        return True
    except Exception:
        return False

def detect_outliers_iqr(df, cols=None):
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
    outlier_counts = {}
    mask = pd.Series(True, index=df.index)
    for col in cols:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if series.empty:
            outlier_counts[col] = 0
            continue
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        col_mask = df[col].between(lower, upper) | df[col].isna()
        outlier_counts[col] = int((~col_mask).sum())
        mask &= col_mask
    return mask, outlier_counts

def df_to_bytes_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def text_to_bytes(s):
    return s.encode('utf-8')

def generate_pdf_report(df_before, df_after, missing_report, outlier_report, notes=None):
    """
    Simple PDF generation using reportlab if available.
    Returns bytes or None.
    """
    if not REPORTLAB_AVAILABLE:
        return None
    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    x = 40
    y = height - 40
    p.setFont("Helvetica-Bold", 14)
    p.drawString(x, y, "SmartSight - Preprocessing Report")
    p.setFont("Helvetica", 10)
    y -= 20
    p.drawString(x, y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y -= 24

    p.setFont("Helvetica-Bold", 12)
    p.drawString(x, y, "Missing Values Summary")
    y -= 16
    p.setFont("Helvetica", 10)
    for col, cnt in missing_report.items():
        p.drawString(x, y, f"{col}: {cnt}")
        y -= 12
        if y < 80:
            p.showPage()
            y = height - 40

    y -= 8
    p.setFont("Helvetica-Bold", 12)
    p.drawString(x, y, "Outlier Detection Summary")
    y -= 16
    p.setFont("Helvetica", 10)
    for col, cnt in outlier_report.items():
        p.drawString(x, y, f"{col}: {cnt} outliers detected")
        y -= 12
        if y < 80:
            p.showPage()
            y = height - 40

    if notes:
        y -= 8
        p.setFont("Helvetica-Bold", 12)
        p.drawString(x, y, "Notes")
        y -= 12
        p.setFont("Helvetica", 10)
        for line in str(notes).splitlines():
            p.drawString(x, y, line[:95])
            y -= 10
            if y < 80:
                p.showPage()
                y = height - 40

    # before/after snapshots
    y -= 8
    p.setFont("Helvetica-Bold", 12)
    p.drawString(x, y, "Before (first 5 rows)")
    y -= 12
    p.setFont("Helvetica", 8)
    for line in df_before.head(5).to_string().splitlines():
        p.drawString(x, y, line[:95])
        y -= 9
        if y < 80:
            p.showPage()
            y = height - 40

    y -= 8
    p.setFont("Helvetica-Bold", 12)
    p.drawString(x, y, "After (first 5 rows)")
    y -= 12
    p.setFont("Helvetica", 8)
    for line in df_after.head(5).to_string().splitlines():
        p.drawString(x, y, line[:95])
        y -= 9
        if y < 80:
            p.showPage()
            y = height - 40

    p.save()
    buffer.seek(0)
    return buffer.getvalue()

def add_audit_log(log_list, msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"{ts} - {msg}"
    log_list.append(entry)
    return log_list

# ------------------ Data loading UI ------------------

st.sidebar.header("📁 Load dataset")
st.sidebar.write("Upload CSV/Excel or paste a CSV URL. You can also load the provided sample dataset.")

uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
url_input = st.sidebar.text_input("OR paste CSV URL (http/https/local path)")

# Use your uploaded file path (developer instruction): default local sample path
default_local_path = "/mnt/data/f74247c1-1511-4973-a3f8-c7c528f76667.csv"
if st.sidebar.checkbox("Load default sample dataset"):
    url_input = default_local_path

df = None
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.success("✅ Uploaded file loaded")
    except Exception as e:
        st.error(f"Unable to read uploaded file: {e}")

elif url_input:
    try:
        # handle local path or URL
        if os.path.exists(url_input):
            df = pd.read_csv(url_input)
        else:
            resp = pd.read_csv(url_input)  # pandas can read many URLs directly
            df = resp
        st.success("✅ Dataset loaded from path/URL")
    except Exception as e:
        st.error(f"Unable to fetch dataset from URL/path: {e}")

if df is None:
    st.info("Please upload a file or provide a URL / select default dataset.")
    st.stop()

# Keep original copy for reports
original_df = df.copy()

# normalize column names
df.columns = df.columns.str.strip()

# ensure object columns are string for Arrow compatibility
for c in df.select_dtypes(include=['object']).columns:
    df[c] = df[c].astype(str)

# audit log
audit_log = []
add_audit_log(audit_log, f"Loaded dataset with shape {df.shape}")

# Basic info display
st.subheader("📋 Dataset Overview")
c1, c2 = st.columns([2, 1])
with c1:
    st.write("Shape:", df.shape)
    st.write("Columns:", df.columns.tolist())
with c2:
    st.write("Preview:")
    st.dataframe(df.head(5))

# Data quality report
st.markdown("---")
st.subheader("📄 Automated Data Quality Report")
missing_counts = df.isnull().sum().to_dict()
missing_percent = (df.isnull().mean() * 100).round(2).to_dict()
unique_counts = df.nunique().to_dict()
dtypes = df.dtypes.apply(lambda x: x.name).to_dict()

report_df = pd.DataFrame({
    "column": df.columns,
    "dtype": [dtypes[c] for c in df.columns],
    "missing_count": [missing_counts[c] for c in df.columns],
    "missing_perc": [missing_percent[c] for c in df.columns],
    "unique_count": [unique_counts[c] for c in df.columns]
})
st.dataframe(report_df)

# allow user to download initial report CSV
st.download_button("Download Data Quality CSV", data=df_to_bytes_csv(report_df), file_name="data_quality_report.csv")

# ------------------ Missing value handling ------------------
st.markdown("---")
st.subheader("🧹 Missing Value Handling")
st.write("Select a strategy and click Apply. A downloadable CSV and PDF (if available) will be provided.")

df_before_missing = df.copy()
missing_action = st.selectbox("Default strategy for numeric/categorical missing values:",
                              ["Median/Mode", "Mean/Mode", "Zero", "Forward Fill", "None"])

apply_missing = st.button("Apply Missing Strategy")

if apply_missing:
    filled_counts = {}
    total_filled = 0
    for col in df.columns:
        cnt_before = int(df_before_missing[col].isnull().sum())
        if cnt_before == 0:
            filled_counts[col] = 0
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            if missing_action == "Median/Mode":
                fill_val = df[col].median()
            elif missing_action == "Mean/Mode":
                fill_val = df[col].mean()
            elif missing_action == "Zero":
                fill_val = 0
            elif missing_action == "Forward Fill":
                df[col].fillna(method='ffill', inplace=True)
                filled_counts[col] = cnt_before - int(df[col].isnull().sum())
                total_filled += filled_counts[col]
                continue
            else:
                filled_counts[col] = 0
                continue
            df[col].fillna(fill_val, inplace=True)
        else:
            # categorical/object
            if missing_action in ["Median/Mode", "Mean/Mode"]:
                try:
                    fill_val = df[col].mode()[0]
                except Exception:
                    fill_val = "Unknown"
                df[col].fillna(fill_val, inplace=True)
            elif missing_action == "Zero":
                df[col].fillna("Unknown", inplace=True)
            elif missing_action == "Forward Fill":
                df[col].fillna(method='ffill', inplace=True)
            else:
                pass
        filled_counts[col] = cnt_before - int(df[col].isnull().sum())
        total_filled += filled_counts[col]

    st.success(f"Missing values handled — total cells filled: {total_filled}")
    st.dataframe(pd.DataFrame.from_dict(filled_counts, orient='index', columns=['filled_count']))

    add_audit_log(audit_log, f"Missing values handled: total_filled={total_filled}")

    # downloads
    st.download_button("Download CSV after Missing Handling", data=df_to_bytes_csv(df),
                       file_name="data_after_missing.csv", mime="text/csv")

    if REPORTLAB_AVAILABLE:
        missing_report = {col: int(df_before_missing[col].isnull().sum()) for col in df.columns}
        pdf_bytes = generate_pdf_report(df_before_missing, df, missing_report, {})
        if pdf_bytes:
            st.download_button("Download PDF (Missing Handling Report)", data=pdf_bytes,
                               file_name="missing_handling_report.pdf", mime="application/pdf")
    else:
        st.info("Install reportlab to enable PDF reports: pip install reportlab")

# ------------------ Outlier detection/removal ------------------
st.markdown("---")
st.subheader("🚨 Outlier Detection & Removal (IQR)")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
st.write("Numeric columns detected:", numeric_cols)
remove_outliers = st.checkbox("Remove outliers (IQR) for numeric columns")

df_before_outliers = df.copy()
outlier_counts = {}
if remove_outliers:
    mask, outlier_counts = detect_outliers_iqr(df, numeric_cols)
    total_outliers = sum(outlier_counts.values())
    df = df[mask]
    st.success(f"Outliers removed — total outlier cells (sum across columns): {total_outliers}")
    st.dataframe(pd.DataFrame.from_dict(outlier_counts, orient='index', columns=['outliers_detected']))
    add_audit_log(audit_log, f"Outlier removal: total_outlier_cells={total_outliers}")

    st.download_button("Download CSV after Outlier Removal", data=df_to_bytes_csv(df),
                       file_name="data_after_outliers.csv", mime="text/csv")

    if REPORTLAB_AVAILABLE:
        pdf_bytes = generate_pdf_report(df_before_outliers, df, {}, outlier_counts)
        if pdf_bytes:
            st.download_button("Download PDF (Outlier Removal Report)", data=pdf_bytes,
                               file_name="outlier_removal_report.pdf", mime="application/pdf")
    else:
        st.info("Install reportlab to enable PDF reports: pip install reportlab")

# ------------------ Encoding ------------------
st.markdown("---")
st.subheader("🔤 Encoding (Categorical)")

categorical_cols = [c for c in df.columns if (df[c].dtype == 'object' or str(df[c].dtype).startswith('category'))]
st.write("Categorical columns detected:", categorical_cols)
encode_choice = st.radio("Choose encoding", ["None", "Label Encoding", "One-Hot Encoding"])

if encode_choice != "None":
    df_before_encoding = df.copy()
    if encode_choice == "Label Encoding":
        le = LabelEncoder()
        applied = []
        for col in categorical_cols:
            try:
                df[col] = le.fit_transform(df[col].astype(str))
                applied.append(col)
            except Exception:
                st.warning(f"Skipping label encoding for {col}")
        st.success(f"Label encoding applied to: {applied}")
        add_audit_log(audit_log, f"Label encoding: cols={applied}")
    else:
        # one-hot
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        st.success("One-hot encoding applied")
        add_audit_log(audit_log, f"One-hot encoding applied to {categorical_cols}")

    st.download_button("Download CSV after Encoding", data=df_to_bytes_csv(df), file_name="data_after_encoding.csv")

# ------------------ Scaling ------------------
st.markdown("---")
st.subheader("📏 Scaling Features")
scale_choice = st.radio("Scaling method", ["None", "StandardScaler", "MinMaxScaler", "RobustScaler"])
if scale_choice != "None":
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        if scale_choice == "StandardScaler":
            scaler = StandardScaler()
        elif scale_choice == "MinMaxScaler":
            scaler = MinMaxScaler()
        else:
            scaler = RobustScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        st.success(f"{scale_choice} applied to numeric columns")
        add_audit_log(audit_log, f"Scaling applied: {scale_choice} on {numeric_cols}")
        st.download_button("Download CSV after Scaling", data=df_to_bytes_csv(df), file_name="data_after_scaling.csv")
    else:
        st.info("No numeric columns to scale")

# ------------------ Visualizations ------------------
st.markdown("---")
st.subheader("📊 Visualizations & Interactive EDA")

plot_type = st.selectbox("Plot type", [
    "None", "Histogram", "Box", "Scatter", "Bar", "Line (with time slicing)", "Correlation Heatmap", "Violin", "Catplot", "Wordcloud"
])

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
text_cols = df.select_dtypes(include=['object']).columns.tolist()
date_cols = [c for c in df.columns if is_datetime_series(original_df[c])]

# helper for line plot time slicing
def parse_datetime_col(name):
    try:
        return pd.to_datetime(df[name], errors='coerce')
    except Exception:
        return None

if plot_type == "Histogram":
    if numeric_cols:
        col = st.selectbox("Select numeric column", numeric_cols)
        bins = st.slider("Bins", 5, 200, 30)
        if col:
            fig, ax = plt.subplots()
            ax.hist(df[col].dropna(), bins=bins, color='skyblue', edgecolor='black')
            ax.set_title(f"Histogram: {col}")
            st.pyplot(fig)
    else:
        st.info("No numeric columns")

elif plot_type == "Box":
    if numeric_cols:
        col = st.selectbox("Select numeric column", numeric_cols)
        if col:
            fig, ax = plt.subplots()
            sns.boxplot(x=df[col].dropna(), ax=ax)
            ax.set_title(f"Boxplot: {col}")
            st.pyplot(fig)
    else:
        st.info("No numeric columns")

elif plot_type == "Scatter":
    if len(numeric_cols) >= 2:
        x = st.selectbox("X-axis", numeric_cols)
        y = st.selectbox("Y-axis", numeric_cols)
        if x and y:
            sample_rate = st.slider("Downsample by factor (1 = all rows)", 1, 50, 1)
            data_plot = df[[x,y]].dropna()
            if sample_rate > 1:
                data_plot = data_plot.iloc[::sample_rate, :]
            fig, ax = plt.subplots()
            ax.scatter(data_plot[x], data_plot[y], alpha=0.6)
            ax.set_xlabel(x); ax.set_ylabel(y)
            ax.set_title(f"Scatter: {x} vs {y}")
            st.pyplot(fig)
    else:
        st.info("Need >=2 numeric columns")

elif plot_type == "Bar":
    if text_cols and numeric_cols:
        x = st.selectbox("Categorical (x-axis)", text_cols)
        y = st.selectbox("Numeric (y-axis)", numeric_cols)
        if x and y:
            grouped = df.groupby(x)[y].sum().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(10,4))
            grouped.plot(kind='bar', ax=ax, color='orange')
            ax.set_title(f"Bar: {y} by {x}")
            st.pyplot(fig)
    else:
        st.info("Need at least one categorical and one numeric column")

elif plot_type == "Line (with time slicing)":
    if date_cols:
        date_col = st.selectbox("Choose datetime column", date_cols)
        # convert copy to datetime
        df['_tmp_timecol'] = pd.to_datetime(original_df[date_col], errors='coerce')
        df_sorted = df.sort_values('_tmp_timecol')
        # allow range selection
        start = st.date_input("Start date", value=df_sorted['_tmp_timecol'].min().date())
        end = st.date_input("End date", value=df_sorted['_tmp_timecol'].max().date())
        # numeric selection and aggregation
        ycol = st.selectbox("Y column (numeric)", numeric_cols)
        agg = st.selectbox("Aggregation", ["sum", "mean", "median"])
        # downsampling/grouping freq
        freq = st.selectbox("Group by", ["D", "W", "M", "Q", "Y"])
        if ycol:
            mask = (df_sorted['_tmp_timecol'].dt.date >= start) & (df_sorted['_tmp_timecol'].dt.date <= end)
            df_period = df_sorted.loc[mask, ['_tmp_timecol', ycol]].dropna()
            if df_period.empty:
                st.info("No data in selected period")
            else:
                df_period = df_period.set_index('_tmp_timecol').resample(freq).agg(agg)
                fig, ax = plt.subplots(figsize=(10,4))
                ax.plot(df_period.index, df_period[ycol], marker='o')
                ax.set_title(f"{ycol} ({agg}) from {start} to {end} grouped by {freq}")
                ax.set_xlabel("Time")
                ax.set_ylabel(ycol)
                st.pyplot(fig)
        # cleanup
        df.drop(columns=['_tmp_timecol'], inplace=True, errors=False)
    else:
        st.info("No datetime-like columns detected. The app auto-detects date-like columns from the original dataset.")

elif plot_type == "Correlation Heatmap":
    if numeric_cols:
        fig, ax = plt.subplots(figsize=(10,6))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='YlGnBu', ax=ax)
        st.pyplot(fig)
    else:
        st.info("No numeric columns")


elif plot_type == "Violin":
    if numeric_cols and text_cols:
        cat = st.selectbox("Categorical column", text_cols)
        num = st.selectbox("Numeric column", numeric_cols)
        fig, ax = plt.subplots(figsize=(8,4))
        sns.violinplot(x=df[cat].astype(str), y=df[num], ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        st.pyplot(fig)
    else:
        st.info("Need categorical and numeric columns")

elif plot_type == "Catplot":
    if text_cols and numeric_cols:
        cat = st.selectbox("Categorical", text_cols)
        num = st.selectbox("Numeric", numeric_cols)
        kind = st.selectbox("Kind", ["bar", "box", "strip", "swarm"])
        sample_size = min(1000, len(df))
        sample_df = df[[cat, num]].dropna().sample(sample_size, random_state=42)
        g = sns.catplot(x=cat, y=num, data=sample_df, kind=kind, height=4, aspect=2)
        st.pyplot(g.fig)
        plt.close(g.fig)
    else:
        st.info("Need categorical and numeric columns")

elif plot_type == "Wordcloud":
    if WORDCLOUD_AVAILABLE and text_cols:
        txt_col = st.selectbox("Text column for wordcloud", text_cols)
        text_data = " ".join(df[txt_col].dropna().astype(str).tolist())
        wc = WordCloud(width=800, height=400, background_color='white').generate(text_data)
        fig, ax = plt.subplots(figsize=(10,4))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
    else:
        st.info("Install wordcloud library or no text columns present")


# ------------------ Final preview, audit log and downloads ------------------
st.markdown("---")
st.subheader("✅ Final Cleaned Dataset & Audit")

st.write("Preview (first 10 rows):")
st.dataframe(df.head(10))

st.download_button("Download Final Cleaned CSV", data=df_to_bytes_csv(df), file_name="cleaned_data.csv")

# audit log display and download
st.markdown("**Preprocessing Audit Log**")
st.write("\n".join(audit_log))
st.download_button("Download Audit Log (txt)", data=text_to_bytes("\n".join(audit_log)), file_name="preprocessing_audit_log.txt")

# final PDF of entire process
if REPORTLAB_AVAILABLE:
    final_pdf = generate_pdf_report(original_df, df, missing_counts, outlier_counts if 'outlier_counts' in locals() else {}, notes="\n".join(audit_log))
    if final_pdf:
        st.download_button("Download Final Preprocessing PDF Report", data=final_pdf, file_name="final_preprocessing_report.pdf")
else:
    st.info("Install reportlab to enable full PDF report generation (optional).")

st.success("Processing complete. Use the downloads above for documentation or further modeling.")

