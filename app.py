# app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai
import os
import sqlite3
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Google Gemini API
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# Download NLTK data
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Initialize Gemini model
model = genai.GenerativeModel('gemini-pro')

# Database operations
def init_db():
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS chats
        (id INTEGER PRIMARY KEY AUTOINCREMENT,
         timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
         user_input TEXT,
         ai_response TEXT,
         sentiment_score REAL,
         sentiment_label TEXT)
    ''')
    conn.commit()
    conn.close()

def save_chat(user_input, ai_response, sentiment_score, sentiment_label):
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO chats (user_input, ai_response, sentiment_score, sentiment_label)
        VALUES (?, ?, ?, ?)
    ''', (user_input, ai_response, sentiment_score, sentiment_label))
    conn.commit()
    conn.close()

def get_chat_history():
    conn = sqlite3.connect('chat_history.db')
    df = pd.read_sql_query("SELECT * FROM chats", conn)
    conn.close()
    return df

# Gemini response generation
def generate_response(prompt):
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return None

# Sentiment analysis function
def analyze_sentiment(text):
    sentiment_scores = sia.polarity_scores(text)
    compound_score = sentiment_scores['compound']
    
    if compound_score >= 0.05:
        sentiment_label = 'Positive'
    elif compound_score <= -0.05:
        sentiment_label = 'Negative'
    else:
        sentiment_label = 'Neutral'
    
    return compound_score, sentiment_label

# Initialize database
init_db()

# Streamlit UI
st.title('Gemini AI Chatbot with Sentiment Analysis')

# Sidebar for analytics
st.sidebar.title('Analytics Dashboard')

# Get chat history
df = get_chat_history()

if not df.empty:
    # Sentiment distribution pie chart
    sentiment_counts = df['sentiment_label'].value_counts()
    fig_pie = px.pie(values=sentiment_counts.values, names=sentiment_counts.index, 
                     title='Sentiment Distribution')
    st.sidebar.plotly_chart(fig_pie)
    
    # Sentiment trend over time
    fig_line = px.line(df, x='timestamp', y='sentiment_score', 
                       title='Sentiment Trend Over Time')
    st.sidebar.plotly_chart(fig_line)
    
    # Summary statistics
    st.sidebar.subheader('Summary Statistics')
    st.sidebar.write(f"Total conversations: {len(df)}")
    st.sidebar.write(f"Average sentiment score: {df['sentiment_score'].mean():.2f}")

# Main chat interface
user_input = st.text_input("You:", key="user_input")

if st.button('Send'):
    if user_input:
        with st.spinner('Generating response...'):
            # Generate AI response
            ai_response = generate_response(user_input)
            
            if ai_response:
                # Analyze sentiment
                sentiment_score, sentiment_label = analyze_sentiment(ai_response)
                
                # Save to database
                save_chat(user_input, ai_response, sentiment_score, sentiment_label)
                
                # Display response
                st.write(f"AI: {ai_response}")
                st.write(f"Sentiment: {sentiment_label} (Score: {sentiment_score:.2f})")

# Display chat history
st.subheader("Chat History")
history = get_chat_history()
if not history.empty:
    for idx, row in history.iloc[::-1].iterrows():
        with st.expander(f"Conversation {idx}", expanded=False):
            st.write(f"Time: {row['timestamp']}")
            st.write(f"You: {row['user_input']}")
            st.write(f"AI: {row['ai_response']}")
            st.write(f"Sentiment: {row['sentiment_label']} (Score: {row['sentiment_score']:.2f})")

# Download chat history
if not df.empty:
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download Chat History as CSV",
        data=csv,
        file_name="chat_history.csv",
        mime="text/csv",
    )