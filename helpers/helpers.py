# Importing important libraries
import os
# import re
import base64
import string
import numpy as np
import pandas as pd
from stqdm import stqdm

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
    

# # Function to clean text (each row of data from dataframe.)
# def clean_text(text):
#     # Convert to lowercase
#     text = text.lower()
    
#     # Remove punctuation using string.punctuation
#     text = text.translate(str.maketrans('', '', string.punctuation))
    
#     # Remove any non-alphanumeric characters (excluding spaces)
#     # text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
#     return text


def extract_top_n_keywords(text_list, vectorizer, top_n=10):
    """Extracts top N keywords from a batch of texts using vectorization."""
    result_matrix = vectorizer.transform(text_list).toarray()
    feature_names = vectorizer.get_feature_names_out()
    result_df = pd.DataFrame(result_matrix, columns=feature_names)
    # print("RESULT STUFF HERE\n\n",result_df)

    return result_df.apply(lambda row: ", ".join(row[row > 0].nlargest(top_n).index.tolist()), axis=1).tolist()


def parallel_extract_topics(df, vectorizer, top_n=5, batch_size=500):
    """Applies batch + parallel processing for topic extraction."""
    n_cores = min(cpu_count(), 4)  # Limit to 4 cores
    text_batches = np.array_split(df["clean_text"], len(df) // batch_size + 1)

    with Pool(n_cores) as pool:
        results = pool.starmap(extract_top_n_keywords, [(batch.tolist(), vectorizer, top_n) for batch in text_batches])

    df["rough_topics"] = [item for sublist in results for item in sublist]
    # print(df["rough_topics"])
    return df


def process_rough_topics_to_meaningful_topic_batch(rough_topics_batch, model="gpt-3.5-turbo"):
    """Converts rough topics into a single meaningful topic using OpenAI."""
    client = OpenAI(api_key=OPEN_AI_KEY)

    system_prompt = """You are an expert topic modeling analyst specializing in converting TF-IDF generated topics into clear, concise, and meaningful topics. Your task is to:

1. Analyze each line of TF-IDF topics carefully
2. Identify the core theme or concept being discussed
3. Generate a single, specific, and informative topic label
4. Return topics in a clean CSV format, one topic per line
5. Ensure each topic is:
   - Specific enough to distinguish it from other topics
   - General enough to be understood without context
   - Consistent in formatting and style
   - Professional and academic in tone

Return only the topics, one per line, with no additional text, numbers, or formatting."""

    user_prompt = """Below are TF-IDF generated topic groups. For each line, provide one clear, specific topic that best represents the underlying theme. Return only the topics in CSV format, one per line:

Example input:
1. machine learning artificial intelligence neural networks
2. climate change global warming environmental impact

Example output:
Deep Learning and Neural Network Applications
Climate Change Environmental Effects

Now analyze these topics:
"""
    
    # Add each topic group to the prompt
    for idx, topics in enumerate(rough_topics_batch):
        user_prompt += f"\n{idx + 1}. {topics}"
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": "I will provide specific, meaningful topics for each line, one per line, in CSV format."},
                {"role": "user", "content": "Proceed with the analysis. Remember to return only the topics, one per line, with no additional text or formatting."}
            ],
            temperature=0.3  # Lower temperature for more consistent outputs
        )
        summary_topic = response.choices[0].message.content.strip()
        return summary_topic.split("\n")

    except Exception as e:
        print(f"Error processing topics: {e}")
        return ["Error generating topic"] * len(rough_topics_batch)


def process_in_batches(df, batch_size=5):
    """Processes rough topics into meaningful topics in batches."""
    all_meaningful_topics = []
    n_rows = len(df)
    
    for start in stqdm(range(0, n_rows, batch_size), desc="Processing batches"):
        end = min(start + batch_size, n_rows)
        rough_topics_batch = df["rough_topics"].iloc[start:end].tolist()
        meaningful_topics_batch = process_rough_topics_to_meaningful_topic_batch(rough_topics_batch)
        all_meaningful_topics.extend(meaningful_topics_batch)
    df["meaningful_topic"] = all_meaningful_topics
    return df