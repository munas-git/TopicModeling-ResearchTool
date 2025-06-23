import time

# Importing important libraries
import streamlit as st

# Data wrangling related.
import re
import pandas as pd

# Data Visualisation related.
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Topic Modelling related.
from multiprocessing import Pool
from sklearn.feature_extraction.text import CountVectorizer

# others
from helpers.helpers import *


def clean_text(text):
    """Function to clean text (modify as needed)."""
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove punctuation using string.punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove any non-alphanumeric characters (excluding spaces)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    return text


def create_vectorizer(n_grams, max_df_threshold):
    """Creates the CountVectorizer."""
    return CountVectorizer(ngram_range=n_grams, stop_words="english", max_df=max_df_threshold)


def extract_topics(df, _vectorizer, top_n=5, batch_size=300):
    """Performs parallel topic extraction."""
    return parallel_extract_topics(df, _vectorizer, top_n, batch_size)


def refine_topics(df, batch_size=300):
    """
    Refines topics using OpenAI.
    
    Args:
        df: DataFrame containing 'Topic' and 'Frequency' columns
        batch_size: Number of topics to process in each batch
    
    Returns:
        DataFrame with refined topics added as 'meaningful_topic' column
    """
    return process_in_batches(df, batch_size)


def process_topic_frequencies(transformed_data, vectorizer):
    """
    Process topic frequencies and refine topics.
    
    Args:
        transformed_data: Sparse matrix of TF-IDF scores
        vectorizer: Fitted TF-IDF vectorizer
    
    Returns:
        DataFrame with refined topics
    """
    # Calculate topic frequencies
    topic_counts = np.asarray(transformed_data.sum(axis=0)).flatten()
    topics = vectorizer.get_feature_names_out()
    topic_df = pd.DataFrame({'Base Topics': topics, 'Frequency': topic_counts})

    # Filter topics above mean frequency
    mean_frequency = topic_df['Frequency'].mean()
    above_mean_df = topic_df[topic_df['Frequency'] >= mean_frequency]
    above_mean_df = above_mean_df.sort_values(by='Frequency', ascending=False).reset_index(drop=True)

    # Refine topics
    processed_df = refine_topics(above_mean_df, batch_size=300)
    
    return processed_df


# Initialize session state variables
if 'uploaded_df' not in st.session_state:
    st.session_state.uploaded_df = None
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'current_file_name' not in st.session_state:
    st.session_state.current_file_name = None


# Set page title
st.set_page_config(page_title="TopicModeling-ResearchTool")

# logo section
# logo_image_base64 = get_base64_image("./images/---.png")
# col1, col2 = st.columns([1, 5])
# with col1:
#     st.markdown(
#         f"""
#         <style>
#         .top-left-logo {{
#             position: relative;
#             top: -50%;
#             left: -180%;
#         }}
#         </style>
#         <img class="top-left-logo" src="data:image/png;base64,{logo_image_base64}" width="130">
#         """,
#         unsafe_allow_html=True
#     )


# Title section
st.markdown(
    """
    <style>
        .centered-title {
            text-align: center;  /* Centers text */
            font-size: 36px;  /* Adjust font size */
            font-weight: bold;  /* Optional: Makes it bold */
        }
    </style>
    <h1 class="centered-title">
        <span style="color:#00843D;">Topic</span> 
        <span style="color:black;">Modelling Tool.</span>
    </h1>
    <br>
    """,
    unsafe_allow_html=True
)


# Upload cvs section
uploaded_file = st.file_uploader("Upload CSV for analysis. (NOTE: each row representing a new document)", type=["csv"])

if uploaded_file is not None:
    # Check if a new file has been uploaded by comparing filenames... couldnt compare uploaded file object directly
    current_file = uploaded_file.name
    if st.session_state.current_file_name != current_file:
        st.session_state.uploaded_df = pd.read_csv(uploaded_file)
        st.session_state.processed_df = None  # Reset processed data
        st.session_state.current_file_name = current_file

    # Display the first few rows of the DataFrame
    st.write("Data Preview:")
    st.dataframe(st.session_state.uploaded_df.head(2), use_container_width=True)

    # Column selection
    if len(st.session_state.uploaded_df.columns) > 1:
        column_name = st.selectbox("Select the column for analysis:", st.session_state.uploaded_df.columns)
    else:
        column_name = st.session_state.uploaded_df.columns[0]

    col2_1, col2_2 = st.columns(2)
    with col2_1:
        n_gram_choice = st.selectbox(
            "Select N-gram range:",
            ("Bi-grams (2-grams)", "Tri-grams (3-grams)", "Four-grams (4-grams)", "Five-grams (5-grams)"),
            help="Bi-grams (2-grams) capture sequences of 2 words. Tri-grams (3-grams) capture sequences of 3 words. Four-grams (4-grams) capture sequences of 4 words. Five-grams (5-grams) capture sequences of 5 words. Choose based on how fine-grained you want the analysis to be"
        )

        n_grams = {
            "Bi-grams (2-grams)": (2, 2),
            "Tri-grams (3-grams)": (3, 3),
            "Four-grams (4-grams)": (4, 4),
            "Five-grams (5-grams)": (5, 5)
        }[n_gram_choice]

    with col2_2:
        input_container = st.container()
        threshold = input_container.number_input(
            "N-gram frequency limit (%)",
            min_value=0,
            max_value=100,
            value=50,
            step=5,
            help="Exclude n-grams appearing in more than X% of documents. A higher value excludes extremely common terms."
        )
        max_df_threshold = threshold / 100.0

    # Check if analysis needs to be rerun
    needs_rerun = (
        st.session_state.processed_df is None or
        st.session_state.current_file_name != current_file
    )

    # Trigger analysis
    status_placeholder = st.empty()
    if st.button("Begin Analysis"):
        if needs_rerun:
            try:
                # Step 1: Cleaning Data
                status_placeholder.text("Step 1: Cleaning Data...")
                time.sleep(1)
                st.session_state.uploaded_df["clean_text"] = st.session_state.uploaded_df[column_name].apply(clean_text)

                # Step 2: Creating and caching vectorizer
                status_placeholder.text("Step 2: Extracting Topics...")
                vectorizer = create_vectorizer(n_grams, max_df_threshold)
                
                # Fit the vectorizer on the entire dataset first
                vectorizer.fit(st.session_state.uploaded_df["clean_text"])
                
                # Transform each chunk in parallel
                with Pool() as pool:
                    chunks = np.array_split(st.session_state.uploaded_df["clean_text"], pool._processes)
                    transformed_chunks = pool.map(vectorizer.transform, chunks)
                
                # Convert each chunk to a dense array and combine them
                transformed_data = np.vstack([chunk.toarray() for chunk in transformed_chunks])
                
                # Process topics and store results
                status_placeholder.text("Step 3: Processing and Refining Topics...")
                st.session_state.processed_df = process_topic_frequencies(
                    transformed_data,
                    vectorizer
                )
                st.session_state.topic_frequencies = st.session_state.processed_df[["Base Topics", "Frequency", "AI Refined Topic"]]

            except ValueError:
                st.error("Please select column containing document abstract/content.")

        status_placeholder.empty()

    # Display Results
    if st.session_state.processed_df is not None:
        st.divider()
        st.markdown("""
            <style>
                .centered-result {
                    text-align: center;
                    font-size: 36px;
                    font-weight: bold;
                }
            </style>
            <h1 class="centered-result">
                <span style="color:#00843D;">Analysis</span> 
                <span style="color:black;">Results.</span>
            </h1>
            <br>
            """, unsafe_allow_html=True)

        st.markdown(f"#### Topics Covered & Their Frequencies.")
        topic_search_term = st.text_input("Search Topics:", "").split(" ")

        if topic_search_term:
            filtered_freq = st.session_state.topic_frequencies[
                st.session_state.topic_frequencies["Base Topics"].apply(
                    lambda x: all(term.lower() in str(x).lower() for term in topic_search_term)
                )
            ]
        else:
            filtered_freq = st.session_state.topic_frequencies

        st.dataframe(filtered_freq, use_container_width=True, height=250)
