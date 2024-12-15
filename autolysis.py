# /// 
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

# Set your OpenAI API token from environment variable
openai.api_key = os.getenv("AIPROXY_TOKEN")  # AIPROXY_TOKEN

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

# Visualize the data and save the plots
def visualize_data(df, output_dir, csv_name):
    """
    Generate and save visualizations for the data (histograms and count plots).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plot_paths = []

    # Create visualizations for numerical data
    for col in df.select_dtypes(include=[np.number]).columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col} in {csv_name}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"{col}_histogram.png")
        plt.savefig(plot_path)
        plot_paths.append(plot_path)
        plt.close()

    # Create visualizations for categorical columns
    for col in df.select_dtypes(include=[np.object]).columns:
        plt.figure(figsize=(8, 6))
        sns.countplot(x=df[col])
        plt.title(f"Count of categories in {col} for {csv_name}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"{col}_categories.png")
        plt.savefig(plot_path)
        plot_paths.append(plot_path)
        plt.close()

    return plot_paths

# Generate a detailed README file
def generate_readme_with_ai(df, plot_paths, csv_name, output_dir):
    """
    Generate a README.md file using OpenAI API.
    """
    num_rows, num_columns = df.shape
    dtype_summary = "\n".join([f"- **{dtype}**: {count} columns" for dtype, count in df.dtypes.value_counts().items()])
    
    # Summary for numerical columns
    numerical_summary = "\n".join(
        [f"- **{col}**: Mean = {df[col].mean():.2f}, Std Dev = {df[col].std():.2f}" for col in df.select_dtypes(include=[np.number]).columns]
    )
    
    # Summary for categorical columns
    categorical_summary = "\n".join(
        [f"- **{col}**: Top categories - {', '.join(map(str, df[col].value_counts().index[:5]))}" for col in df.select_dtypes(exclude=[np.number]).columns]
    )
    
    prompt = f"""
    You are a professional data analysis assistant. Generate a Markdown README for a dataset analysis named "{csv_name}".
    The dataset contains {num_rows} rows and {num_columns} columns.
    - Data types breakdown:
    {dtype_summary}
    - Numerical columns insights:
    {numerical_summary}
    - Categorical columns insights:
    {categorical_summary}
    - Visualizations have been generated and saved. List of plots:
    {', '.join(plot_paths)}
    
    Provide a professional summary, including any insights or conclusions drawn from this analysis.
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # Ensure correct model version
            messages=[{"role": "system", "content": "You are a professional data analysis assistant."},
                      {"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.7
        )
        readme_content = response['choices'][0]['message']['content'].strip()
    except Exception as e:
        readme_content = f"Error generating README: {e}\nFallback prompt:\n{prompt}"

    readme_path = os.path.join(output_dir, 'README.md')
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    print(f"README.md created successfully at {readme_path}")

# Main function to process datasets
def process_datasets():
    """
    Process all CSV files in the current directory.
    """
    csv_files = [f for f in os.listdir() if f.endswith('.csv')]
    if not csv_files:
        print("No CSV files found in the current directory.")
        return

    for csv_file in csv_files:
        try:
            print(f"Processing {csv_file}...")
            df = pd.read_csv(csv_file)
            df = clean_and_process_columns(df)
            base_name = os.path.splitext(os.path.basename(csv_file))[0]
            output_dir = os.path.join("output", base_name)
            os.makedirs(output_dir, exist_ok=True)
            plot_paths = visualize_data(df, output_dir, base_name)
            generate_readme_with_ai(df, plot_paths, csv_file, output_dir)
            print(f"Analysis for {csv_file} completed successfully.")
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            traceback.print_exc()

# Run the script
if __name__ == "__main__":
    process_datasets()
