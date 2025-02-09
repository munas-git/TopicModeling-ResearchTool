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
from sklearn.feature_extraction.text import CountVectorizer

# others
from helpers.helpers import *

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
    uploaded_file = pd.read_csv(uploaded_file)

    # Display the first few rows of the DataFrame
    st.write("Data Preview:")
    st.dataframe(uploaded_file.head(2), use_container_width = True)

    # Step 3: Allow the user to select a column
    if len(uploaded_file.columns) > 1:
        column_name = st.selectbox("Select the column for analysis:", uploaded_file.columns)
    else:
        column_name = uploaded_file.columns[0]
    

    col2_1, col2_2 = st.columns(2)
    with col2_1:
        # Step 4: Select n-gram range using selectbox
        n_gram_choice = st.selectbox(
            "Select N-gram range:",
            ("Tri-grams (3-grams)", "Four-grams (4-grams)", "Five-grams (5-grams)"),
            help="Tri-grams (3-grams) capture sequences of 3 words. Four-grams (4-grams) capture sequences of 4 words. Five-grams (5-grams) capture sequences of 5 words. Choose based on how fine-grained you want the analysis to be"
        )

        # Set the n-gram range based on user selection
        if n_gram_choice == "Tri-grams (3-grams)":
            n_grams = (3, 3)
        elif n_gram_choice == "Four-grams (4-grams)":
            n_grams = (4, 4)
        else:
            n_grams = (5, 5)

    with col2_2:
        input_container = st.container()
        # Add number input and tooltip in the same line
        threshold = input_container.number_input(
            "N-gram frequency limit (%)",
            min_value=0, 
            max_value=100, 
            value=50,
            step=5,
            help="Exclude n-grams appearing in more than X% of documents. A higher value excludes extremely common terms."
        )
        max_df_threshold = threshold / 100.0  # Convert percentage to a fraction


    # Trigger analysis
    status_placeholder = st.empty()
    if st.button("Begin Analysis"):
        
        # cleaning with cleaning function
        status_placeholder.text("Step 1: Cleaning Data...")
        time.sleep(2) #################################################
        uploaded_file["clean_text"] = uploaded_file[column_name].apply(clean_text)

        # extracting topics
        status_placeholder.text("Step 2: Extracting Topics...")
        time.sleep(2) #################################################
        topics_vectorizer = CountVectorizer(ngram_range=n_grams, stop_words="english", max_df=max_df_threshold)
        X_topics = topics_vectorizer.fit_transform(uploaded_file["clean_text"])

        top_n = 5  # Specify the number of top keywords to extract
        uploaded_file["rough_topics"] = uploaded_file["clean_text"].apply(lambda x: extract_top_n_keywords_as_string(x, topics_vectorizer, top_n))
        status_placeholder.text("Step 3: Raw Topics Extracted...")

        time.sleep(2)
        status_placeholder.text("Step 4: Processing Raw Topics in Batches...")
        uploaded_file = process_in_batches(uploaded_file, batch_size=500)
        status_placeholder.empty()


        ####################### Result Display Section ##########################
        st.divider()
        st.markdown(
            """
            <style>
                .centered-result {
                    text-align: center;  /* Centers text */
                    font-size: 36px;  /* Adjust font size */
                    font-weight: bold;  /* Optional: Makes it bold */
                }
            </style>
            <h1 class="centered-result">
                <span style="color:#00843D;">Analysis</span> 
                <span style="color:black;">Results.</span>
            </h1>
            <br>
            """,
            unsafe_allow_html=True
        )

        st.markdown(f"#### Topics Covered.")
        # searchable dataframe for event reviews
        topic_search_term = st.text_input("Search Topics:", "").split(" ")

        # Filter DataFrame based on search term
        if topic_search_term:
            uploaded_file_filtered = uploaded_file[uploaded_file["meaningful_topic"].apply(lambda x: all(term.lower() in str(x).lower() for term in topic_search_term))]
        else:
            uploaded_file_filtered = uploaded_file
        st.dataframe(uploaded_file["meaningful_topic"], use_container_width = True, height = 200)