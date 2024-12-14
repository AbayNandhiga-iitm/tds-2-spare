# /// script
# requires-python = ">=3.8"
# dependencies = [
#   "pandas>=1.1.0",
#   "matplotlib>=3.3.0",
#   "seaborn>=0.11.0",
#   "requests>=2.25.0",
#   "openai>=0.27.0"
# ]
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests

# Ensure the API token is loaded
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
if not AIPROXY_TOKEN:
    raise EnvironmentError("AIPROXY_TOKEN is not set. Please configure it as an environment variable.")

# Base URL for AI Proxy
AI_PROXY_BASE_URL = "https://aiproxy.sanand.workers.dev/openai/v1"

# Function to attempt reading CSV with different encodings
def load_csv_with_encoding(csv_file):
    encodings = ['utf-8', 'ISO-8859-1', 'latin1', 'utf-16']  # Try common encodings
    for encoding in encodings:
        try:
            return pd.read_csv(csv_file, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise Exception("Unable to read CSV file with supported encodings.")

# Function to clean and process meaningful columns
def clean_columns(df):
    df = df.loc[:, ~df.columns.str.contains('id|url|isbn|image|name', case=False, na=False)]
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() > 50:  # Drop high-cardinality categorical columns
            df = df.drop(columns=[col])
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna(how='all')
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category')
    return df

# Function to generate a story using AI Proxy's GPT model
def generate_story_with_llm(summary):
    try:
        url = f"{AI_PROXY_BASE_URL}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {AIPROXY_TOKEN}"
        }
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are an insightful data analyst narrating findings."},
                {"role": "user", "content": summary}
            ]
        }
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        return f"Error generating story with LLM: {e}"

# Function to generate analysis and visualizations
def analyze_and_visualize(csv_file):
    try:
        df = load_csv_with_encoding(csv_file)
        dataset_name = os.path.splitext(os.path.basename(csv_file))[0]
        df = clean_columns(df)
        output_dir = dataset_name
        os.makedirs(output_dir, exist_ok=True)

        description = df.describe(include="all").to_string()
        missing_values = df.isnull().sum().to_string()

        heatmap_path = "No correlation heatmap generated (insufficient numerical data)."
        if df.select_dtypes(include=['number']).shape[1] > 1:
            plt.figure(figsize=(10, 8))
            sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
            plt.title("Correlation Heatmap")
            heatmap_path = os.path.join(output_dir, "heatmap.png")
            plt.savefig(heatmap_path)
            plt.close()

        distribution_path = "No distribution plot generated (no numeric data)."
        numeric_columns = df.select_dtypes(include=['number']).columns
        if len(numeric_columns) > 0:
            top_feature = numeric_columns[0]
            plt.figure(figsize=(8, 6))
            sns.histplot(df[top_feature], kde=True, color='blue')
            plt.title(f"Distribution of {top_feature}")
            distribution_path = os.path.join(output_dir, "distribution.png")
            plt.savefig(distribution_path)
            plt.close()

        summary = (
            f"Dataset Name: {dataset_name}\n\n"
            f"Basic Description:\n{description}\n\n"
            f"Missing Values:\n{missing_values}\n\n"
            f"Generated Visualizations:\n"
            f"1. Correlation Heatmap: {heatmap_path}\n"
            f"2. Distribution of {top_feature if len(numeric_columns) > 0 else 'N/A'}: {distribution_path}\n"
        )

        story = generate_story_with_llm(summary)

        readme_path = os.path.join(output_dir, "README.md")
        with open(readme_path, "w") as readme_file:
            readme_file.write(summary + "\n\n" + story)

        print(f"Analysis completed. Outputs saved in: {output_dir}")
    except Exception as e:
        print(f"Error during analysis: {e}")

# Main execution
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_story.py <path_to_csv>")
    else:
        csv_file = sys.argv[1]
        analyze_and_visualize(csv_file)
