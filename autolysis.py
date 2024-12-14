# /// script
# requires-python = ">=3.8"
# dependencies = [
#   "pandas>=1.3.0",
#   "matplotlib>=3.4.0",
#   "seaborn>=0.11.0",
#   "openai>=0.27.0"
# ]
# ///

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import traceback

# --- FUNCTIONS ---

def load_csv_with_encoding(csv_file):
    """Load a CSV file with fallback encoding handling."""
    encodings = ['utf-8', 'ISO-8859-1', 'latin1', 'utf-16']
    for encoding in encodings:
        try:
            return pd.read_csv(csv_file, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise Exception(f"Unable to read {csv_file} with common encodings.")

def clean_and_process_columns(df):
    """Clean the dataset dynamically based on column types."""
    # Drop high-cardinality columns
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() > 50:
            df = df.drop(columns=[col])
    
    # Convert applicable columns to numeric and drop completely empty columns
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna(how='all', axis=1)  # Drop columns with all NaNs

    return df

def generate_visualizations(df, output_dir, dataset_name):
    """Generate visualizations dynamically for any dataset."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    visualizations = []

    # Pairplot for numerical data
    numeric_data = df.select_dtypes(include=['number'])
    if not numeric_data.empty:
        plt.figure(figsize=(10, 8))
        sns.pairplot(numeric_data, diag_kind='kde')
        pairplot_path = os.path.join(output_dir, f"{dataset_name}_pairplot.png")
        plt.savefig(pairplot_path)
        plt.close()
        visualizations.append(pairplot_path)

    # Boxplot for categorical vs numerical data
    for cat_col in df.select_dtypes(include=['object', 'category']).columns:
        for num_col in numeric_data.columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=cat_col, y=num_col, data=df)
            boxplot_path = os.path.join(output_dir, f"{dataset_name}_boxplot_{cat_col}_{num_col}.png")
            plt.savefig(boxplot_path)
            plt.close()
            visualizations.append(boxplot_path)

    return visualizations

def summarize_dataset(df):
    """Generate a textual summary of the dataset."""
    summary = f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.\n\n"
    summary += "### Column Details:\n"
    for col in df.columns:
        summary += f"- `{col}`: {df[col].dtype} ({df[col].nunique()} unique values)\n"
    return summary

def generate_readme(dataset_name, output_dir, summary, visualizations):
    """Generate a README.md file for the dataset."""
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w") as readme_file:
        readme_file.write(f"# Analysis for {dataset_name}\n\n")
        readme_file.write("## Summary\n\n")
        readme_file.write(summary + "\n\n")
        if visualizations:
            readme_file.write("## Visualizations\n\n")
            for vis in visualizations:
                vis_rel_path = os.path.relpath(vis, output_dir)
                readme_file.write(f"![{os.path.basename(vis_rel_path)}]({vis_rel_path})\n\n")

def run_analysis(dataset_path):
    """Perform dynamic analysis on any dataset."""
    try:
        dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
        output_dir = os.path.join("output", dataset_name)
        
        # Step 1: Load Dataset
        df = load_csv_with_encoding(dataset_path)
        processed_df = clean_and_process_columns(df)
        
        # Step 2: Generate Visualizations
        visualizations = generate_visualizations(processed_df, output_dir, dataset_name)

        # Step 3: Generate README
        summary = summarize_dataset(df)
        generate_readme(dataset_name, output_dir, summary, visualizations)
        
        print(f"Analysis completed for {dataset_name}. Results saved in {output_dir}.")
    
    except Exception as e:
        traceback.print_exc()
        print(f"Error processing dataset {dataset_path}: {str(e)}")

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    datasets = [f for f in os.listdir('.') if f.endswith('.csv')]
    if datasets:
        for dataset in datasets:
            run_analysis(dataset)
    else:
        print("No CSV datasets found in the current directory.")
