# /// script
# requires-python = ">=3.8"
# dependencies = [
#   "pandas>=1.1.0",
#   "matplotlib>=3.3.0",
#   "seaborn>=0.11.0",
#   "wordcloud>=1.8.0",
#   "scikit-learn>=0.24.0",
#   "requests>=2.25.0",
#   "openai>=0.27.0"
# ]
# ///

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from wordcloud import WordCloud
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Ensure the API token is loaded
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
if not AIPROXY_TOKEN:
    raise EnvironmentError("AIPROXY_TOKEN is not set. Please configure it as an environment variable.")

# Base URL for AI Proxy
AI_PROXY_BASE_URL = "https://aiproxy.sanand.workers.dev/openai/v1"

def load_csv_with_encoding(csv_file: str) -> pd.DataFrame:
    """Attempt to load a CSV file using multiple encodings."""
    encodings = ['utf-8', 'ISO-8859-1', 'latin1', 'utf-16']
    for encoding in encodings:
        try:
            return pd.read_csv(csv_file, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unable to read CSV file: {csv_file} with supported encodings.")

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and process meaningful columns in the DataFrame."""
    df = df.loc[:, ~df.columns.str.contains('id|url|isbn|image|name', case=False, na=False)]
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() > 50:
            df = df.drop(columns=[col])
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna(how='all')
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category')
    return df

def generate_wordcloud(text: str, output_dir: str) -> str:
    """Generate a word cloud from a text column."""
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    wc_path = os.path.join(output_dir, "wordcloud.png")
    wordcloud.to_file(wc_path)
    return wc_path

def cluster_and_visualize(df: pd.DataFrame, output_dir: str) -> str:
    """Perform clustering and visualize results."""
    numeric_data = df.select_dtypes(include=['number']).dropna()
    if numeric_data.empty:
        return "No clustering performed (insufficient numerical data)."
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_data)
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)
    numeric_data['Cluster'] = clusters
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=numeric_data.iloc[:, 0], y=numeric_data.iloc[:, 1], hue=clusters, palette="viridis")
    plt.title("Clustering Results")
    cluster_path = os.path.join(output_dir, "clustering.png")
    plt.savefig(cluster_path)
    plt.close()
    return cluster_path

def generate_story_with_llm(summary: str) -> str:
    """Generate a story using GPT."""
    try:
        url = f"{AI_PROXY_BASE_URL}/chat/completions"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {AIPROXY_TOKEN}"}
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are an insightful data analyst creating a compelling report."},
                {"role": "user", "content": summary}
            ]
        }
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        logging.error(f"Error generating story: {e}")
        return "Error generating story."

def analyze_and_visualize(csv_file: str):
    """Main function for analysis and visualization."""
    try:
        df = load_csv_with_encoding(csv_file)
        dataset_name = os.path.splitext(os.path.basename(csv_file))[0]
        output_dir = dataset_name
        os.makedirs(output_dir, exist_ok=True)
        df = clean_columns(df)

        # Generate statistical summary
        description = df.describe(include="all").to_string()
        missing_values = df.isnull().sum().to_string()

        # Generate visualizations
        heatmap_path = "No heatmap generated (insufficient data)."
        if df.select_dtypes(include=['number']).shape[1] > 1:
            plt.figure(figsize=(10, 8))
            sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
            plt.title("Correlation Heatmap")
            heatmap_path = os.path.join(output_dir, "heatmap.png")
            plt.savefig(heatmap_path)
            plt.close()

        wordcloud_path = generate_wordcloud(" ".join(df.select_dtypes(include=['object']).fillna("").stack()), output_dir)

        cluster_path = cluster_and_visualize(df, output_dir)

        # Compile summary
        summary = (
            f"Dataset Name: {dataset_name}\n\n"
            f"Description:\n{description}\n\n"
            f"Missing Values:\n{missing_values}\n\n"
            f"Visualizations:\nHeatmap: {heatmap_path}\nWord Cloud: {wordcloud_path}\nClustering: {cluster_path}"
        )

        story = generate_story_with_llm(summary)
        readme_path = os.path.join(output_dir, "README.md")
        with open(readme_path, "w") as readme_file:
            readme_file.write(summary + "\n\n" + story)
        logging.info(f"Analysis completed. Outputs saved in: {output_dir}")
    except Exception as e:
        logging.error(f"Error during analysis: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        logging.error("Usage: python analyze_story.py <path_to_csv>")
    else:
        analyze_and_visualize(sys.argv[1])
