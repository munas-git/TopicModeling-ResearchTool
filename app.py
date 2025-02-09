import time

# Importing important libraries
import streamlit as st

# Data wrangling related.
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


# Function caching
@st.cache_data(show_spinner=False)
def clean_text(text):
    """Function to clean text (modify as needed)."""
    return text.lower().strip()

@st.cache_resource(show_spinner=False)
def create_vectorizer(n_grams, max_df_threshold):
    """Creates and caches the CountVectorizer."""
    return CountVectorizer(ngram_range=n_grams, stop_words="english", max_df=max_df_threshold)

@st.cache_data(show_spinner=False)
def extract_topics(df, _vectorizer, top_n=5, batch_size=300):  # Added underscore to vectorizer
    """Performs parallel topic extraction and caches results."""
    return parallel_extract_topics(df, _vectorizer, top_n, batch_size)

@st.cache_data(show_spinner=False)
def refine_topics(df, batch_size=300):
    """Refines topics using OpenAI and caches results."""
    return process_in_batches(df, batch_size)

# Initialize session state variables
if 'uploaded_df' not in st.session_state:
    st.session_state.uploaded_df = None
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'current_column' not in st.session_state:
    st.session_state.current_column = None
if 'current_ngram' not in st.session_state:
    st.session_state.current_ngram = None
if 'current_threshold' not in st.session_state:
    st.session_state.current_threshold = None
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'current_file_name' not in st.session_state:
    st.session_state.current_file_name = None


# Set page title
st.set_page_config(page_title="CABI Topic Modelling Tool")

# logo section
logo_image_base64 = get_base64_image("./images/CabiKnowledgeLogo.png")
col1, col2 = st.columns([1, 5])
with col1:
    st.markdown(
        f"""
        <style>
        .top-left-logo {{
            position: relative;
            top: -50%;
            left: -180%;
        }}
        </style>
        <img class="top-left-logo" src="data:image/png;base64,{logo_image_base64}" width="130">
        """,
        unsafe_allow_html=True
    )


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
uploaded_file = st.file_uploader("Upload CSV for analysis.", type=["csv"])

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
            ("Tri-grams (3-grams)", "Four-grams (4-grams)", "Five-grams (5-grams)"),
            help="Tri-grams (3-grams) capture sequences of 3 words. Four-grams (4-grams) capture sequences of 4 words. Five-grams (5-grams) capture sequences of 5 words. Choose based on how fine-grained you want the analysis to be"
        )

        n_grams = {
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
        st.session_state.current_column != column_name or
        st.session_state.current_ngram != n_grams or
        st.session_state.current_threshold != max_df_threshold
    )

    # Trigger analysis
    status_placeholder = st.empty()
    if st.button("Begin Analysis"):
        if needs_rerun:
            # Step 1: Cleaning Data
            status_placeholder.text("Step 1: Cleaning Data...")
            time.sleep(1)
            st.session_state.uploaded_df["clean_text"] = st.session_state.uploaded_df[column_name].apply(clean_text)

            # Step 2: Creating and caching vectorizer
            status_placeholder.text("Step 2: Extracting Topics...")
            st.session_state.vectorizer = create_vectorizer(n_grams, max_df_threshold)
            st.session_state.vectorizer.fit(st.session_state.uploaded_df["clean_text"])

            # Step 3: Parallelized Topic Extraction
            status_placeholder.text("Step 3: Assigning Topics...")
            st.session_state.processed_df = extract_topics(
                st.session_state.uploaded_df,
                st.session_state.vectorizer,
                top_n=5,
                batch_size=300
            )

            # Step 4: Refining Topics
            status_placeholder.text("Step 4: Refining Topics in Batch...")
            st.session_state.processed_df = refine_topics(st.session_state.processed_df, batch_size=500)
            # print("PROCESSED FILE",st.session_state.processed_df)

            # Update current parameters
            st.session_state.current_column = column_name
            st.session_state.current_ngram = n_grams
            st.session_state.current_threshold = max_df_threshold

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

        st.markdown(f"#### Topics Covered.")
        topic_search_term = st.text_input("Search Topics:", "").split(" ")

        if topic_search_term:
            filtered_df = st.session_state.processed_df[
                st.session_state.processed_df["meaningful_topic"].apply(
                    lambda x: all(term.lower() in str(x).lower() for term in topic_search_term)
                )
            ]
        else:
            filtered_df = st.session_state.processed_df
        st.dataframe(filtered_df["meaningful_topic"], use_container_width=True, height=250)