import streamlit as st
import faiss
import numpy as np
import pandas as pd
import cohere
from datetime import datetime
import os
from google_play_scraper import app
from dotenv import load_dotenv

load_dotenv()


# Initialize Cohere client
cohere_api_key = os.getenv('api')  # Replace with your Cohere API key
co = cohere.Client(cohere_api_key)

# Load the FAISS index from a file
index = faiss.read_index("faiss_index.bin")

# Load the DataFrame
csv_file_path = r'C:\Users\Dell\3D Objects\NLP\gg\nowgg_embeddings.csv'  # Replace with the path to your CSV file
test3 = pd.read_csv(csv_file_path)

# Function to get embedding for a query using Cohere
def get_query_embedding(query):
    response = co.embed(texts=[query])
    return np.array(response.embeddings[0][:250]).astype('float32')

# Function to perform similarity search
def search_similar(query, k=5):
    query_embedding = get_query_embedding(query).reshape(1, -1)
    distances, indices = index.search(query_embedding, k)
    
    results = []
    for idx in indices[0]:
        product_id = test3.iloc[idx]['product_id']
        # Fetch app details from Google Play Store
        app_details = app(product_id)
        result = {
            'title': test3.iloc[idx]['title'],
            'product_id': test3.iloc[idx]['product_id'],
            'description': test3.iloc[idx]['final_description'],
            'link': test3.iloc[idx]['link'],
            #'icon':app_details["icon"]
            'video':app_details["video"]
        }
        results.append(result)
    return results

# Function to save feedback
def save_feedback(query, feedback):
    feedback_data = {
        'timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        'query': [query],
        'feedback': [feedback]
    }
    feedback_df = pd.DataFrame(feedback_data)
    
   
    feedback_df.to_csv('feedback.csv', mode='a', header=False, index=False)

    
path=r"C:\Users\Dell\3D Objects\NLP\game.jpg"    
# HTML & CSS for the app
st.markdown("""
    <style>
    body {
        font-family: 'Arial', sans-serif;
    
        }
    .title {
        font-size: 2.5em;
        color: #4CAF50;
        text-align: center;
        margin-bottom: 20px;
    }
    .query-input {
        text-align: center;
        margin-bottom: 20px;
    }
    .result-card {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .result-title {
        font-size: 1.5em;
        color: #333;
        margin-bottom: 10px;
    }
    .result-productid {
        font-size: 1.0em;
        color: #333;
        margin-bottom: 5px;
    }
    .result-link {
        color: #0066cc;
        text-decoration: none;
    }
    .result-link:hover {
        text-decoration: underline;
    }
    .feedback-section {
        margin-top: 40px;
        text-align: center;
    }
    .feedback-textarea {
        width: 100%;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #ccc;
        margin-bottom: 20px;
    }
    .submit-btn {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
    }
    .submit-btn:hover {
        background-color: #45a049;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit app
st.markdown('<div class="title">Game Recommendation System</div>', unsafe_allow_html=True)

query = st.text_input("Enter your query:", key="query_input", placeholder="Type something...")
if query:
    top_k_results = search_similar(query)
    st.write('<div class="query-input">Top recommendations:</div>', unsafe_allow_html=True)
    
    for result in top_k_results:
        #img=result["product_id"]
        st.markdown(f"""
            <div class="result-card">
                <div class="result-title">{result['title']}</div>
                <div><a class="result-link" href="{result['link']}">Link</a></div>
            </div>
        """, unsafe_allow_html=True)
        
        st.video(result['video'])
        #video_url=result['video']  # Display the image
        
        


        
    st.markdown('<div class="feedback-section">################ Feedback #####################</div>', unsafe_allow_html=True)
    feedback = st.text_area("Please provide your feedback here:", key="feedback_textarea", height=100)
    if st.button("Submit Feedback", key="submit_feedback"):
        save_feedback(query, feedback)
        st.write("Thank you for your feedback!")

# Run the app with:
# streamlit run hello.py
#<div class="result-description">**Description**: {result['description']}</div>
#<div class="result-productid">**Product-id**{result['product_id']}</div>
