# Importing important libraries
import os
# import re
import base64
import string
import pandas as pd
from tqdm import tqdm

# LLM Related
from openai import OpenAI

# others
from dotenv import load_dotenv
load_dotenv()

OPEN_AI_KEY = os.getenv("OPEN_AI_KEY")


# Function to get base64 image (for logo)
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()
    

# Function to clean text (each row of data from dataframe.)
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation using string.punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove any non-alphanumeric characters (excluding spaces)
    # text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    return text


def extract_top_n_keywords_as_string(text, vectorizer, top_n=10):
    # Transform the text (the text input must be in a list, as the vectorizer works on a list of documents)
    result = vectorizer.transform([text])
    result = result.toarray()

    # Get feature names (n-grams)
    feature_names = vectorizer.get_feature_names_out()

    # Create a dictionary of n-grams and their frequencies
    decoded_result = {}
    for idx, value in enumerate(result[0]):
        if value > 0:  # Only include n-grams that appear in the text
            decoded_result[feature_names[idx]] = value

    # Sort the n-grams by frequency (descending) and take the top N
    sorted_keywords = sorted(decoded_result.items(), key=lambda x: x[1], reverse=True)[:top_n]

    # Format the result as a string
    keyword_strings = [f"'{keyword}' appeared {count} times" for keyword, count in sorted_keywords]
    return ', '.join(keyword_strings)


def process_rough_topics_to_meaningful_topic_batch(rough_topics_batch, model="gpt-3.5-turbo"):
    client = OpenAI(api_key=OPEN_AI_KEY)
    
    # Create a prompt for the batch
    prompt = "Generate a single, meaningful topic that best summarizes these topics:\n"
    for idx, topics in enumerate(rough_topics_batch):
        prompt += f"\n{idx + 1}. {topics}"
    
    try:
        # Make the API call to OpenAI's GPT model for the batch
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. For each group of topics, provide a single concise topic summary."},
                {"role": "user", "content": prompt}
            ]
        )
        
        # Get the summary topic from response
        summary_topic = response.choices[0].message.content.strip()
        
        # Return the same summary for all items in the batch
        return [summary_topic] * len(rough_topics_batch)
        
    except Exception as e:
        print(f"Error processing topics: {e}")
        return ["Error generating topic"] * len(rough_topics_batch)


def process_in_batches(df, batch_size=5):  # Reduced batch size for better processing
    all_meaningful_topics = []
    n_rows = len(df)
    
    # Process in smaller batches
    for start in tqdm(range(0, n_rows, batch_size), desc="Processing batches"):
        end = min(start + batch_size, n_rows)
        rough_topics_batch = df["rough_topics"].iloc[start:end].tolist()
        # print("## \nRough topics batch: ", rough_topics_batch)
        
        # Get meaningful topics for the batch
        meaningful_topics_batch = process_rough_topics_to_meaningful_topic_batch(rough_topics_batch)
        # print("## \n\nMeaningful topics batch: ", meaningful_topics_batch)
        
        # Append the results to the list
        all_meaningful_topics.extend(meaningful_topics_batch)
    
    # Add the meaningful topics to the dataframe
    df["meaningful_topic"] = all_meaningful_topics
    return df