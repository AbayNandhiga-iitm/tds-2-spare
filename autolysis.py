# /// script
# requires-python = ">=3.8"
# dependencies = [
#   "pandas",
#   "matplotlib",
#   "seaborn",
#   "openai"
# ]
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from openai import ChatCompletion
import sys

# Ensure AIPROXY_TOKEN is set as an environment variable
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
if not AIPROXY_TOKEN:
    sys.exit("AIPROXY_TOKEN is not set. Please set it as an environment variable.")

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
    # Automatically drop columns that are likely identifiers (like ISBN, URLs, IDs, etc.)
    # We drop columns that have non-numeric values and seem to be identifiers
    df = df.loc[:, ~df.columns.str.contains('id|url|isbn|image|name', case=False, na=False)]

    # Remove columns with excessive unique values (e.g., book titles, URLs) that don't add value
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() > 50:  # Heuristic: Drop high-cardinality categorical columns
            df = df.drop(columns=[col])

    # Automatically convert columns to numeric where applicable
    df = df.apply(pd.to_numeric, errors='coerce')

    # Drop rows where all values are NaN
    df = df.dropna(how='all')

    # Convert remaining object columns to 'category'
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category')

    return df

# Function to generate an analysis and visualizations
def analyze_and_visualize(csv_file):
    try:
        # Load the dataset with different encoding options
        df = load_csv_with_encoding(csv_file)
        dataset_name = os.path.splitext(os.path.basename(csv_file))[0]

        # Clean the dataset to focus on meaningful columns
        df = clean_columns(df)

        # Create a directory for the output
        output_dir = dataset_name
        os.makedirs(output_dir, exist_ok=True)

        # Basic insights
        description = df.describe(include="all").to_string()
        missing_values = df.isnull().sum().to_string()

        # Visualization 1: Correlation heatmap for numerical data
        if df.select_dtypes(include=['number']).shape[1] > 1:
            plt.figure(figsize=(10, 8))
            sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
            plt.title("Correlation Heatmap")
            heatmap_path = os.path.join(output_dir, "heatmap.png")
            plt.savefig(heatmap_path)
            plt.close()
        else:
            heatmap_path = "No correlation heatmap generated (insufficient numerical data)."

        # Visualization 2: Distribution of top numeric feature
        numeric_columns = df.select_dtypes(include=['number']).columns
        if len(numeric_columns) > 0:
            top_feature = numeric_columns[0]
            plt.figure(figsize=(8, 6))
            sns.histplot(df[top_feature], kde=True, color='blue')
            plt.title(f"Distribution of {top_feature}")
            distribution_path = os.path.join(output_dir, "distribution.png")
            plt.savefig(distribution_path)
            plt.close()
        else:
            distribution_path = "No distribution plot generated (no numeric data)."

        # Summarize analysis
        summary = (
            f"Dataset Name: {dataset_name}\n\n"
            f"Basic Description:\n{description}\n\n"
            f"Missing Values:\n{missing_values}\n\n"
            f"Generated Visualizations:\n"
            f"1. Correlation Heatmap: {heatmap_path}\n"
            f"2. Distribution of {top_feature}: {distribution_path}\n"
        )

        # Use LLM to narrate the story
        story = generate_story_with_llm(summary)

        # Create README file
        readme_path = os.path.join(output_dir, "README.txt")
        with open(readme_path, "w") as readme_file:
            readme_file.write(summary + "\n\n" + story)

        print(f"Analysis completed. Outputs saved in: {output_dir}")
    except Exception as e:
        print(f"Error during analysis: {e}")

# Function to generate a story using OpenAI's GPT
def generate_story_with_llm(summary):
    try:
        response = ChatCompletion.create(
            model="gpt-4",
            messages=[ 
                {"role": "system", "content": "You are an analyst narrating insights from a dataset."},
                {"role": "user", "content": summary}
            ],
            api_key=AIPROXY_TOKEN  # Use the new AIPROXY_TOKEN for authentication
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Error generating story with LLM: {e}"

# Main execution
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_story.py <path_to_csv>")
    else:
        csv_file = sys.argv[1]
        analyze_and_visualize(csv_file)
