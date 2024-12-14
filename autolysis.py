# /// Script
# requires-python = ">=3.8"
# dependencies = [
#   "pandas>=1.3.0",
#   "seaborn>=0.11.0",
#   "matplotlib>=3.4.0",
#   "numpy>=1.20.0",
#   "scipy>=1.7.0",
#   "openai>=0.27.0"
# ]
# ///
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import openai
import traceback

# Set your OpenAI API token
openai.api_key = os.getenv("AIPROXY_TOKEN")  # Replace with your token or set via environment variable

# Load CSV with encoding handling
def load_csv_with_encoding(csv_file):
    encodings = ['utf-8', 'ISO-8859-1', 'latin1', 'utf-16']
    for encoding in encodings:
        try:
            return pd.read_csv(csv_file, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise Exception(f"Unable to read {csv_file} with common encodings.")

# Clean and process columns
def clean_and_process_columns(df):
    # Remove high-cardinality object columns
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() > 50:
            df = df.drop(columns=[col])

    # Convert columns to numeric where possible
    df = df.apply(pd.to_numeric, errors='coerce')

    # Drop columns and rows with all NaN values
    df = df.dropna(how='all', axis=1)
    df = df.dropna(how='all', axis=0)

    # Remove outliers using z-scores
    z_scores = stats.zscore(df.select_dtypes(include=['number']), nan_policy='omit')
    df = df[(np.abs(z_scores) < 3).all(axis=1)]
    return df

# Helper function to check column meaningfulness
def is_meaningful_column(column, df, unique_min=0.1, unique_max=0.9):
    unique_ratio = df[column].nunique() / len(df[column])
    return unique_min < unique_ratio < unique_max

# Analyze categorical columns
def analyze_categorical_column(column, df, top_n=5):
    return df[column].value_counts().head(top_n)

# Generate visualizations
def generate_visualizations(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    plot_paths = []

    # Numerical column histograms
    numerical_columns = [col for col in df.select_dtypes(include=[np.number]).columns if is_meaningful_column(col, df)]
    for column in numerical_columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[column], kde=True, bins=30, color='teal')
        plt.title(f"Histogram of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plot_path = os.path.join(output_dir, f"{column}_histogram.png")
        plt.savefig(plot_path)
        plot_paths.append(plot_path)
        plt.close()

    # Categorical column count plots
    categorical_columns = [col for col in df.select_dtypes(exclude=[np.number]).columns if is_meaningful_column(col, df)]
    for column in categorical_columns:
        plt.figure(figsize=(8, 6))
        sns.countplot(
            data=df, 
            y=column, 
            order=df[column].value_counts().index[:5], 
            palette='viridis'
        )
        plt.title(f"Top Categories in {column}")
        plt.ylabel(column)
        plt.xlabel("Count")
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plot_path = os.path.join(output_dir, f"{column}_categories.png")
        plt.savefig(plot_path)
        plot_paths.append(plot_path)
        plt.close()

    return plot_paths

# Generate README using OpenAI
def generate_readme_with_ai(output_dir, df, plot_paths):
    num_rows, num_columns = df.shape
    dtype_summary = "\n".join([f"- **{dtype}**: {count} columns" for dtype, count in df.dtypes.value_counts().items()])
    numerical_summary = "\n".join(
        [f"- **{col}**: Mean = {df[col].mean():.2f}, Std Dev = {df[col].std():.2f}" for col in df.select_dtypes(include=[np.number]).columns]
    )
    categorical_summary = "\n".join(
        [f"- **{col}**: Top categories - {', '.join(map(str, analyze_categorical_column(col, df).index))}" for col in df.select_dtypes(exclude=[np.number]).columns]
    )

    prompt = f"""
    You are an AI tasked with creating a Markdown README for a dataset analysis.
    - Dataset contains {num_rows} rows and {num_columns} columns.
    - Data types breakdown:
    {dtype_summary}
    - Numerical columns insights:
    {numerical_summary}
    - Categorical columns insights:
    {categorical_summary}
    - Visualizations were generated for key columns.

    Write a professional README.md summarizing these insights and linking the visualizations.
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are a professional data analysis assistant."},
                      {"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.7
        )
        readme_content = response['choices'][0]['message']['content'].strip()
    except Exception as e:
        readme_content = f"Error generating README: {e}\nFallback:\n{prompt}"

    readme_path = os.path.join(output_dir, 'README.md')
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    print(f"README.md created successfully at {readme_path}")

# Main function to process datasets
def process_datasets():
    csv_files = [f for f in os.listdir() if f.endswith('.csv')]
    for csv_file in csv_files:
        try:
            df = load_csv_with_encoding(csv_file)
            df = clean_and_process_columns(df)
            base_name = os.path.splitext(os.path.basename(csv_file))[0]
            output_dir = os.path.join("output", base_name)
            os.makedirs(output_dir, exist_ok=True)
            plot_paths = generate_visualizations(df, output_dir)
            generate_readme_with_ai(output_dir, df, plot_paths)
            print(f"Analysis for {csv_file} completed successfully.")
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            traceback.print_exc()

# Run the script
if __name__ == "__main__":
    process_datasets()
