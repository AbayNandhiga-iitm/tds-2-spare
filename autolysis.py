# Dependencies (metadata)
# dependencies = [
#   "pandas>=1.3.0",
#   "seaborn>=0.11.0",
#   "matplotlib>=3.4.0",
#   "numpy>=1.20.0",
#   "scipy>=1.7.0",
#   "openai>=0.27.0"
# ]

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import openai
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import traceback

# Set your OpenAI API token
openai.api_key = os.getenv("AIPROXY_TOKEN")  # Use the environment variable AIPROXY_TOKEN

# Load CSV with encoding handling
def load_csv_with_encoding(csv_file):
    """
    Load CSV file with support for multiple encodings.
    """
    encodings = ['utf-8', 'ISO-8859-1', 'latin1', 'utf-16']
    for encoding in encodings:
        try:
            return pd.read_csv(csv_file, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise Exception(f"Unable to read {csv_file} with common encodings.")

# Clean and process columns
def clean_and_process_columns(df):
    """
    Clean the DataFrame by handling missing values, removing high-cardinality columns,
    converting numeric data, and removing outliers.
    """
    # Remove high-cardinality object columns
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() > 50:
            df = df.drop(columns=[col])

    # Convert columns to numeric where possible
    df = df.apply(pd.to_numeric, errors='coerce')

    # Drop columns and rows with all NaN values
    df = df.dropna(axis=1, how='all').dropna(axis=0, how='all')

    # Remove outliers using Z-score
    z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
    df = df[(z_scores < 3).all(axis=1)]
    
    return df

# Visualize the data
def visualize_data(df, output_dir, csv_name):
    """
    Generate and save visualizations for the data.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create visualizations for numerical data
    for col in df.select_dtypes(include=[np.number]).columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col} in {csv_name}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{col}_histogram.png"))
        plt.close()

    # Create visualizations for categorical data
    for col in df.select_dtypes(exclude=[np.number]).columns:
        plt.figure(figsize=(8, 6))
        sns.countplot(x=col, data=df)
        plt.title(f"Countplot of {col} in {csv_name}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{col}_countplot.png"))
        plt.close()

# Generate dynamic README.md
def generate_dynamic_readme(output_dir, df, plot_paths, csv_name):
    """
    Generate a detailed README.md for the dataset analysis.
    """
    num_rows, num_columns = df.shape
    numerical_summary = "\n".join(
        [f"- **{col}**: Mean = {df[col].mean():.2f}, Std Dev = {df[col].std():.2f}" for col in df.select_dtypes(include=[np.number]).columns]
    )
    categorical_summary = "\n".join(
        [f"- **{col}**: Top categories - {', '.join(map(str, analyze_categorical_column(col, df).index))}" for col in df.select_dtypes(exclude=[np.number]).columns]
    )

    correlation_matrix = df.corr()
    correlation_summary = f"Correlation matrix:\n{correlation_matrix}"

    # Dynamic prompt creation for AI-generated README
    prompt = f"""
    Based on the dataset "{csv_name}", the following insights were derived:
    - The dataset contains {num_rows} rows and {num_columns} columns.
    - A summary of numerical features: {numerical_summary}
    - Key findings from categorical data analysis: {categorical_summary}
    - Correlation analysis insights: {correlation_summary}
    - Visualizations: {', '.join([os.path.basename(path) for path in plot_paths])} are included to support these findings.
    Provide a detailed explanation of the key insights, supported by the data and visualizations.
    """

    try:
        # Fallback to GPT-3.5 if GPT-4 is unavailable
        response = openai.Completion.create(
            model="gpt-4",  # Priority use of GPT-4
            prompt=prompt,
            max_tokens=1500,
            temperature=0.7
        )
    except openai.error.InvalidRequestError:  # Fallback case if GPT-4 is unavailable
        response = openai.Completion.create(
            model="gpt-3.5-turbo",  # Use GPT-3.5 if GPT-4 is unavailable
            prompt=prompt,
            max_tokens=1500,
            temperature=0.7
        )
    
        print("Fallback to GPT-3.5 due to GPT-4 unavailability.")

    readme_content = response.choices[0].text.strip()

    # Save the README.md file
    with open(os.path.join(output_dir, "README.md"), 'w') as f:
        f.write(readme_content)

# Analyze categorical columns
def analyze_categorical_column(col, df):
    """
    Analyze and return the top categories for a categorical column.
    """
    return df[col].value_counts().head(5)

# Main Function to Process and Analyze Dataset
def main(csv_file):
    try:
        # Load data
        df = load_csv_with_encoding(csv_file)
        
        # Clean data
        df_clean = clean_and_process_columns(df)
        
        # Create output directory for visualizations
        output_dir = os.path.join(os.getcwd(), "output")
        plot_paths = visualize_data(df_clean, output_dir, os.path.basename(csv_file))
        
        # Generate detailed README.md
        generate_dynamic_readme(output_dir, df_clean, plot_paths, os.path.basename(csv_file))
        
        print(f"Processing of {csv_file} is complete.")
    except Exception as e:
        print(f"Error processing {csv_file}: {e}")
        print(traceback.format_exc())

# Example usage
if __name__ == "__main__":
    # Example CSV file for testing
    csv_file = "path_to_your_data.csv"  # Replace with the actual path
    main(csv_file)

