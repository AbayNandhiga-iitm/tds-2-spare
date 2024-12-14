# ///script
# requires-python = ">=3.8"
# dependencies = [
#   "pandas>=1.3.0",
#   "seaborn>=0.11.0",
#   "openai>=0.27.0"
# ]
# ///

import os
import pandas as pd
import seaborn as sns
from scipy import stats
import traceback

# Function to automatically load CSV files with encoding handling
def load_csv_with_encoding(csv_file):
    """
    Load a CSV file with encoding handling.
    Tries multiple encodings to read the file.
    """
    encodings = ['utf-8', 'ISO-8859-1', 'latin1', 'utf-16']
    for encoding in encodings:
        try:
            return pd.read_csv(csv_file, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise Exception(f"Unable to read {csv_file} with common encodings.")

# Function to clean and process columns dynamically
def clean_and_process_columns(df):
    """
    Clean and process the dataset based on column types.
    Removes columns with high cardinality, processes missing data, and removes outliers.
    """
    # Remove columns with too many unique values (categorical columns with high cardinality)
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() > 50:
            df = df.drop(columns=[col])

    # Convert remaining columns to numeric and remove those that can't be converted
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna(how='all', axis=1)
    
    # Detect and remove outliers using Z-score for numerical columns
    z_scores = stats.zscore(df.select_dtypes(include=['number']))
    df = df[(z_scores < 3).all(axis=1)]
    
    return df

# Function to create visualizations (Boxplots)
def generate_visualizations(df, dataset_name):
    """
    Generate boxplot visualizations for the dataset.
    """
    # Create a directory to save the visualizations
    output_dir = f'output/{dataset_name}'
    os.makedirs(output_dir, exist_ok=True)

    # Boxplot for categorical vs numerical data (if present)
    for cat_col in df.select_dtypes(include=['object']).columns:
        for num_col in df.select_dtypes(include=['number']).columns:
            boxplot_file = os.path.join(output_dir, f'boxplot_{cat_col}_vs_{num_col}.png')
            sns.boxplot(x=cat_col, y=num_col, data=df)
            sns.savefig(boxplot_file)
            sns.plt.close()

# Function to generate a detailed README.md
def generate_readme(df, dataset_name, output_dir):
    """
    Generate a README.md file summarizing the dataset and including visualizations.
    """
    readme_path = os.path.join(output_dir, 'README.md')
    with open(readme_path, 'w') as f:
        f.write(f"# Analysis of {dataset_name}\n\n")
        f.write(f"**Number of rows**: {df.shape[0]}\n")
        f.write(f"**Number of columns**: {df.shape[1]}\n\n")
        f.write("## Column Details:\n")
        for col in df.columns:
            f.write(f"- **{col}**: {df[col].dtype}, missing values: {df[col].isna().sum()}\n")
        f.write("\n## Visualizations:\n")
        for cat_col in df.select_dtypes(include=['object']).columns:
            for num_col in df.select_dtypes(include=['number']).columns:
                f.write(f"### Boxplot: {cat_col} vs {num_col}:\n")
                f.write(f"![Boxplot](boxplot_{cat_col}_vs_{num_col}.png)\n")
        f.write("\n## Notes:\n")
        f.write("The analysis includes basic visualizations and summary statistics of the dataset.\n")

# Function to process and analyze datasets
def process_datasets():
    """
    Process all CSV files in the current directory, perform analysis, and generate reports.
    """
    # List all CSV files in the current directory
    csv_files = [f for f in os.listdir() if f.endswith('.csv')]
    
    for csv_file in csv_files:
        try:
            # Load the dataset
            df = load_csv_with_encoding(csv_file)
            
            # Clean and process the dataset
            df = clean_and_process_columns(df)
            
            # Create a directory for this dataset
            dataset_name = os.path.splitext(csv_file)[0]
            output_dir = f'output/{dataset_name}'
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate visualizations and save them
            generate_visualizations(df, dataset_name)
            
            # Generate a README.md file summarizing the analysis
            generate_readme(df, dataset_name, output_dir)

            print(f"Analysis for {csv_file} completed successfully.")

        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            traceback.print_exc()

# Run the script
if __name__ == "__main__":
    process_datasets()

