%%writefile appmain.py

#Project flow have been shown from STEP1-------------->STEP9 

#importing the required libraries
import streamlit as st
import faiss
import numpy as np
import pandas as pd
import cohere
from datetime import datetime
import os
from google_play_scraper import app
import firebase_admin
from firebase_admin import credentials, db



# Initialize Cohere client
cohere_api_key = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'  # Replace this with  Cohere API key or load form .env file.I have removed my api key.
co = cohere.Client(cohere_api_key)


# Load the FAISS index from a file
index = faiss.read_index("faiss_index.bin")


# Load the DataFrame
csv_file_path = r'C:\Users\Dell\3D Objects\NLP\nowgg_embeddings.csv'  # Replace with the path to your CSV file
test3 = pd.read_csv(csv_file_path)


if not firebase_admin._apps:
    cred = credentials.Certificate("put here your json file")  # Replace with the path to your Firebase JSON file
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'put here real time database url'
    })




# Function to get embedding for a query using Cohere

def get_query_embedding(query):
    response = co.embed(texts=[query])
    return np.array(response.embeddings[0][:250]).astype('float32')#taking only first 250 features-->converting them to array of float32 type




# Function to perform similarity search and taking top 5 values from it

def search_similar(query, k=5):#STEP 5:Search for similiar query
    
    query_embedding = get_query_embedding(query).reshape(1, -1)#STEP 6:query embedding is obtained(only first 250 features) and is reshaped
    
    distances, indices = index.search(query_embedding, k)#STEP 7:do index search of those 5 similiar embeddings for query and obtain indices
    
    results = []
    
    #iterate through each indices and obtain (title,product_id,final_description,link,video) of those 5 games.
    
    for idx in indices[0]:
        product_id = test3.iloc[idx]['product_id']    #obtain product id
        
        # Fetch app details from Google Play Store using above obtained product id
        app_details = app(product_id)
        
        result = {
            'title': test3.iloc[idx]['title'],
            'product_id': test3.iloc[idx]['product_id'],
            'description': test3.iloc[idx]['final_description'],
            'link': test3.iloc[idx]['link'],
            'video':app_details["video"]
        }
        results.append(result)
    return results         #return results to top_k_results that contain 5 games information




# Function to save feedback
def save_feedback_to_firebase(query, feedback):
    ref = db.reference('provide here node name')
    feedback_data = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'query': query,
        'feedback': feedback
    }
    ref.push(feedback_data)

    
    
# HTML & CSS for the app---------------------------------
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
#-------------------------------------------------------


# Streamlit app
st.markdown('<div class="title">Game Recommendation System</div>', unsafe_allow_html=True)


#STEP 1:user enters the query
query = st.text_input("Enter your query:", key="query_input", placeholder="Type something......")


#STEP 2:if query is found in text-box it will enter into if condition otherwise not
if query:
    top_k_results = search_similar(query)#STEP 3:go to similiar search for the query in (search_similiar) function
    
    st.write('<div class="query-input">Top recommendations:</div>', unsafe_allow_html=True)
    
    for result in top_k_results:#STEP 8:Now obtain the title,link,video of the games
        st.markdown(f"""
            <div class="result-card">
                <div class="result-title">{result['title']}</div>
                <div><a class="result-link" href="{result['link']}">Link</a></div>
            </div>
        """, unsafe_allow_html=True)
        
        st.video(result['video'])
        
#I took only (title, link, video) of each 5 games from top_k_results.I could have obtained other info as well


#feedback mechanisms        
    st.markdown('<div class="feedback-section">################ Feedback #####################</div>', unsafe_allow_html=True)
    feedback = st.text_area("Please provide your feedback here:", key="feedback_textarea", height=100)
    if st.button("Submit Feedback", key="submit_feedback"):
        save_feedback_to_firebase(query, feedback)
        st.write("Thank you for your feedback!")

        
