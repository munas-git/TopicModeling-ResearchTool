# Importing important libraries
import os
import base64
import string
import numpy as np
import pandas as pd
from stqdm import stqdm
from collections import Counter

# LLM Related
from openai import OpenAI

# others
from dotenv import load_dotenv
load_dotenv()
from multiprocessing import Pool, cpu_count

OPEN_AI_KEY = os.getenv("OPEN_AI_KEY")


# Function to get base64 image (for logo)
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()
    

def extract_top_n_keywords(text_list, vectorizer, top_n=10):
    """Extracts top N keywords from a batch of texts using vectorization."""
    result_matrix = vectorizer.transform(text_list).toarray()
    feature_names = vectorizer.get_feature_names_out()
    result_df = pd.DataFrame(result_matrix, columns=feature_names)
    # print("RESULT STUFF HERE\n\n",result_df)

    return result_df.apply(lambda row: ", ".join(row[row > 0].nlargest(top_n).index.tolist()), axis=1).tolist()


def parallel_extract_topics(df, vectorizer, top_n=5, batch_size=300):
    """Applies batch + parallel processing for topic extraction."""
    n_cores = min(cpu_count(), 4)  # Limit to 4 cores
    text_batches = np.array_split(df["clean_text"], len(df) // batch_size + 1)

    with Pool(n_cores) as pool:
        results = pool.starmap(extract_top_n_keywords, [(batch.tolist(), vectorizer, top_n) for batch in text_batches])

    df["rough_topics"] = [item for sublist in results for item in sublist]
    return df


def parallel_extract_topics(df, vectorizer, top_n=5, batch_size=300):
    """Applies batch + parallel processing for topic extraction."""
    n_cores = min(cpu_count(), 4)  # Limit to 4 cores
    text_batches = np.array_split(df["clean_text"], len(df) // batch_size + 1)

    with Pool(n_cores) as pool:
        results = pool.starmap(extract_top_n_keywords, [(batch.tolist(), vectorizer, top_n) for batch in text_batches])

    df["rough_topics"] = [item for sublist in results for item in sublist]
    return df


def process_rough_topics_to_meaningful_topic_batch(topics_batch, model="gpt-3.5-turbo"):
    """
    Converts topics into meaningful topics using OpenAI.
    
    Args:
        topics_batch: List of topics to process
        model: OpenAI model to use
    
    Returns:
        List of meaningful topics
    """
    client = OpenAI(api_key=OPEN_AI_KEY)

    system_prompt = """You are an expert topic modeling analyst specializing in converting TF-IDF generated topics into clear, concise, and meaningful topics. Your task is to:

1. Analyze each topic term
2. Identify the core theme or concept
3. Generate a single, specific, and informative topic label
4. Return topics in a clean CSV format, one topic per line
5. Ensure each topic is:
   - Specific enough to distinguish it from other topics
   - General enough to be understood without context
   - Consistent in formatting and style
   - Professional and academic in tone

Return only the topics, one per line, with no additional text, numbers, or formatting."""

    user_prompt = "Below are topics to analyze. Provide one clear, specific topic for each. Return only the topics, one per line:\n\n"
    
    # Add each topic to the prompt
    for topic in topics_batch:
        user_prompt += f"{topic}\n"
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )
        meaningful_topics = response.choices[0].message.content.strip().split('\n')
        
        # Ensure we return the same number of topics as input
        if len(meaningful_topics) != len(topics_batch):
            meaningful_topics = meaningful_topics[:len(topics_batch)]
            if len(meaningful_topics) < len(topics_batch):
                meaningful_topics.extend(["Topic Generation Error"] * (len(topics_batch) - len(meaningful_topics)))
        
        return meaningful_topics

    except Exception as e:
        print(f"Error processing topics: {e}")
        return ["Topic Generation Error"] * len(topics_batch)


def process_in_batches(df, batch_size=300):
    """
    Processes topics into meaningful topics in batches.
    
    Args:
        df: DataFrame containing 'Topic' and 'Frequency' columns
        batch_size: Number of topics to process in each batch
    
    Returns:
        DataFrame with refined topics added as 'meaningful_topic' column
    """
    all_meaningful_topics = []
    n_rows = len(df)
    
    for start in stqdm(range(0, n_rows, batch_size), desc="Processing batches"):
        end = min(start + batch_size, n_rows)
        # Convert topics to string format suitable for prompt
        topics_batch = df["Topic"].iloc[start:end].tolist()
        meaningful_topics_batch = process_rough_topics_to_meaningful_topic_batch(topics_batch)
        all_meaningful_topics.extend(meaningful_topics_batch)
    
    # Add meaningful topics as a new column
    df_result = df.copy()
    df_result["meaningful_topic"] = all_meaningful_topics
    return df_result