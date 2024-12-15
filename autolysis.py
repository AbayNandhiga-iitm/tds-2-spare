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

# Set OpenAI API token from environment variable
openai.api_key = os.getenv("AIPROXY_TOKEN")

# Clean and process columns with extended statistical analysis
def clean_and_process_columns(df):
    """
    Clean the DataFrame by handling missing values, removing high-cardinality columns,
    converting numeric data, removing outliers, and adding basic statistical features.
    """
    # Remove high-cardinality object columns (>50 unique values)
    for col in df.select_dtypes(include=['object', 'category']).columns:
        if df[col].nunique() > 50:
            df.drop(columns=[col], inplace=True)

    # Convert columns to numeric where possible
    df = df.apply(pd.to_numeric, errors='coerce')

    # Drop columns and rows with all NaN values
    df.dropna(axis=1, how='all', inplace=True)
    df.dropna(axis=0, how='all', inplace=True)

    # Remove outliers using Z-score for numerical columns
    numerical_cols = df.select_dtypes(include=[np.number])
    if not numerical_cols.empty:
        z_scores = np.abs(stats.zscore(numerical_cols, nan_policy='omit'))
        df = df[(z_scores < 3).all(axis=1)]

    # Add statistical summary features for numerical columns
    for col in numerical_cols.columns:
        df[f"{col}_mean"] = numerical_cols[col].mean()
        df[f"{col}_std"] = numerical_cols[col].std()
        df[f"{col}_median"] = numerical_cols[col].median()

    return df

# Visualize the data with enhanced annotations and aesthetics
def visualize_data(df, output_dir, csv_name):
    """
    Generate and save visualizations for the data (histograms for numerical columns, count plots for categorical columns),
    with annotations and improved aesthetics.
    """
    os.makedirs(output_dir, exist_ok=True)
    plot_paths = []

    # Histograms for numerical data
    for col in df.select_dtypes(include=[np.number]).columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[col], kde=True, color='skyblue')
        mean_val = df[col].mean()
        plt.axvline(mean_val, color='red', linestyle='--', label=f"Mean: {mean_val:.2f}")
        plt.title(f"Distribution of {col} in {csv_name}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"{col}_histogram.png")
        plt.savefig(plot_path, dpi=300)
        plot_paths.append(plot_path)
        plt.close()

    # Count plots for categorical data
    for col in df.select_dtypes(include=['object', 'category']).columns:
        plt.figure(figsize=(8, 6))
        sns.countplot(data=df, x=col, palette="Set2")
        plt.title(f"Count of categories in {col} for {csv_name}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"{col}_categories.png")
        plt.savefig(plot_path, dpi=300)
        plot_paths.append(plot_path)
        plt.close()

    return plot_paths

# Generate README using OpenAI dynamically with deeper insights
def generate_readme_with_ai(df, plot_paths, csv_name, output_dir):
    """
    Generate a README.md file using OpenAI API with dynamic prompts based on data insights and deeper statistical analysis.
    """
    num_rows, num_columns = df.shape
    dtype_summary = "\n".join([f"- **{dtype}**: {count} columns" for dtype, count in df.dtypes.value_counts().items()])

    # Summarize numerical columns
    numerical_summary = "\n".join(
        [f"- **{col}**: Mean = {df[col].mean():.2f}, Std Dev = {df[col].std():.2f}, Min = {df[col].min():.2f}, Max = {df[col].max():.2f}"
         for col in df.select_dtypes(include=[np.number]).columns]
    )

    # Summarize categorical columns
    categorical_summary = "\n".join(
        [f"- **{col}**: Top categories - {', '.join(map(str, df[col].value_counts().index[:5]))}"
         for col in df.select_dtypes(include=['object', 'category']).columns]
    )

    # Include additional insights dynamically
    additional_insights = "\n".join(
        [f"The column **{col}** shows significant variance, with a mean of {df[col].mean():.2f} and standard deviation of {df[col].std():.2f}."
         for col in df.select_dtypes(include=[np.number]).columns if df[col].std() > df[col].mean() * 0.5]
    )

    prompt = f"""
    Generate a detailed and professional Markdown README for a dataset analysis named "{csv_name}".
    
    Dataset Summary:
    - **Number of Rows**: {num_rows}
    - **Number of Columns**: {num_columns}

    ### Data Types Breakdown
    {dtype_summary}

    ### Numerical Columns Insights
    {numerical_summary}

    ### Categorical Columns Insights
    {categorical_summary}

    ### Additional Insights
    {additional_insights}

    ### Visualizations
    - {', '.join(plot_paths)}

    Summarize key insights and potential use cases based on this data.
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
        readme_content = f"Error generating README: {e}\nFallback prompt:\n{prompt}"

    readme_path = os.path.join(output_dir, 'README.md')
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    print(f"README.md created successfully at {readme_path}")

# Main processing function with optimized handling for large datasets
def process_datasets():
    """
    Process all CSV files in the current directory with memory-efficient handling for large datasets.
    """
    csv_files = [f for f in os.listdir() if f.endswith('.csv')]
    if not csv_files:
        print("No CSV files found in the current directory.")
        return

    for csv_file in csv_files:
        try:
            print(f"Processing {csv_file}...")
            # Use chunking for large datasets
            chunks = pd.read_csv(csv_file, chunksize=10000)
            df = pd.concat(chunks, ignore_index=True)

            df = clean_and_process_columns(df)
            base_name = os.path.splitext(os.path.basename(csv_file))[0]
            output_dir = os.path.join("output", base_name)
            os.makedirs(output_dir, exist_ok=True)

            # Generate visualizations
            plot_paths = visualize_data(df, output_dir, base_name)

            # Generate README
            generate_readme_with_ai(df, plot_paths, csv_file, output_dir)

            print(f"Analysis for {csv_file} completed successfully.")
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            traceback.print_exc()

# Run the script
if __name__ == "__main__":
    process_datasets()
