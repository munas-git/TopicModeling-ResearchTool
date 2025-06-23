# Topic Modeling-ResearchTool

## Overview
The **Topic Modelling Tool** is a Streamlit-based application designed for extracting and refining topics from textual data. The tool allows users to upload CSV files, select the desired text column for analysis, and extract meaningful topics using NLP techniques. It also integrates OpenAI's API to refine extracted topics into clearer, more interpretable themes.

---

## 🎥 Demo Video (Click to watch ⬇)  
[![Watch the video](https://img.youtube.com/vi/za5Z0IyRAaU/maxresdefault.jpg)](https://youtu.be/za5Z0IyRAaU)  

---

## Features
- **Upload CSV Files**: Users can upload a CSV file containing textual data.
- **Text Cleaning**: Removes punctuation, converts text to lowercase, and eliminates non-alphanumeric characters.
- **N-gram Selection**: Users can choose between bi-grams, tri-grams, four-grams, and five-grams.
- **Frequency Filtering**: Users can set a threshold to exclude highly frequent n-grams.
- **Parallel Processing**: Uses multiprocessing for efficient computation.
- **Topic Extraction**: Generates topics using CountVectorizer from scikit-learn.
- **Topic Refinement**: Uses OpenAI API to convert raw topic terms into meaningful descriptions.
- **Search and View Results**: Users can search for specific topics and view topic frequencies.

---

## Installation
### **Prerequisites**
Ensure you have Python installed (>=3.10.3).

### **1. Clone the Repository**

```bash
git clone https://github.com/munas-git/TopicModeling-ResearchTool.git
cd TopicModelling
```

### **2. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **3. Set Up API Key**
Create a `.env` file in the root directory and add:
```env
OPEN_AI_KEY=your_openai_api_key
```

### **4. Run the Application**
```bash
streamlit run app.py
```

---

## **Preprocessing Steps**
The following steps are applied to clean and prepare the text data before topic extraction:

1. **Text Normalization**
   - Converts text to lowercase.
   - Removes punctuation using `string.punctuation`.
   - Eliminates non-alphanumeric characters (except spaces) using regex.

2. **N-gram Vectorization**
   - Users select an `n-gram range` (e.g., bi-grams, tri-grams, etc.).
   - `CountVectorizer` extracts term frequencies while removing common English stopwords.

3. **Frequency Filtering**
   - Users define a threshold (`max_df_threshold`) to exclude extremely frequent n-grams.
   - This prevents common phrases from dominating the results.

4. **Topic Extraction**
   - The `parallel_extract_topics` function processes text in batches using `multiprocessing.Pool`.
   - The extracted terms are stored as **Base Topics** with their respective frequencies.

5. **Topic Refinement (LLM-based)**
   - Topics are processed in batches via OpenAI’s GPT-3.5/4.
   - The model converts raw n-gram-based topics into **AI Refined Topics**.
   - Ensures topics are specific, meaningful, and easily interpretable.

---

## **User Controls & Inputs**
The tool provides the following controls:

1. **File Upload**
   - Users upload a CSV file with textual content.

2. **Column Selection**
   - If multiple columns exist, users select the column for topic modeling.

3. **N-gram Selection**
   - Options: Bi-grams (2-grams), Tri-grams (3-grams), Four-grams (4-grams), Five-grams (5-grams).

4. **Frequency Limit (%)**
   - Users specify a threshold (0-100%) to exclude common phrases.

5. **Start Analysis**
   - Clicking the "Begin Analysis" button initiates the pipeline.

6. **Search Topics**
   - Users can search extracted topics by entering keywords.

---

## **Final Output & Results**
After analysis, the tool generates:

- **DataFrame Display**: Shows extracted topics, their frequencies, and refined versions.
- **Searchable Topic List**: Users can filter topics using keywords.
- **Download Option** (Future Enhancement): Export results as a CSV file.

---

## **Example Workflow**
1. **Upload a CSV file** containing documents or abstracts.
2. **Select the text column** to analyze.
3. **Choose n-gram settings** and frequency threshold.
4. **Click “Begin Analysis”** to extract topics.
5. **View and search** refined topics in the results table.

---

## **Future Enhancements**
- **Downloadable Reports**: Export topic analysis as CSV/JSON.
- **Interactive Visualizations**: Word clouds and bar charts for topic distribution.
- **Model Customization**: Allow users to choose different NLP models.
- **Cloud Storage Integration**: Store analysis results for later retrieval.
