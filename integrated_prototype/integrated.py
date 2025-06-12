import streamlit as st
import folium
import geopandas as gpd
import pandas as pd
from streamlit_folium import st_folium
import requests
import json

# ---------- App Configuration ----------
st.set_page_config(layout="wide")
st.title('🗺️ Integrated Chat with Map')

# ---------- Groq AI Configuration ----------
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = "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"

def get_groq_response(prompt):
    """
    Function to get response from Groq AI API
    Args:
        prompt (str): User's input message
    Returns:
        str: AI's response
    """
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "messages": [{
            "role": "user",
            "content": prompt
        }]
    }
    
    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error: {str(e)}"

# ---------- Load and Plot Map ----------
try:
    my_subset = gpd.read_file("integrated_prototype/potholes_data.gpkg", layer="potholes")
    small_df = my_subset.head(10)

    def plot_from_df(df, folium_map):
        """
        Function to plot markers on the map
        Args:
            df (DataFrame): Data containing coordinates
            folium_map (Map): Folium map object
        Returns:
            Map: Updated map with markers
        """
        for i, row in df.iterrows():
            folium.Marker(
                location=[row.Latitude, row.Longitude],
                tooltip=f"Location {i}",
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(folium_map)
        return folium_map

    center = [29.358488, -98.626591]
    zoom_start = 10
    m = plot_from_df(small_df, folium.Map(location=center, zoom_start=zoom_start))

    # ---------- Layout ----------
    col1, col2 = st.columns([3, 1])

    # -- Main Map Area --
    with col1:
        st.markdown("### Pothole Map")
        st_data = st_folium(m, width=900, height=500)

    # -- Chatbot Sidebar --
    with col2:
        st.markdown("### Chatbot")
        messages = st.container(height=300)
        
        # Initialize chat history in session state
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat history
        for message in st.session_state.messages:
            with messages.chat_message(message["role"]):
                st.write(message["content"])

        # Chat input and response handling
        if prompt := st.chat_input("Ask about potholes or road conditions"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with messages.chat_message("user"):
                st.write(prompt)

            # Get response from Groq AI
            response = get_groq_response(prompt)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            with messages.chat_message("assistant"):
                st.write(response)

except FileNotFoundError:
    st.error("Error: potholes_data.gpkg file not found. Please ensure the file is in the correct location.")
    st.info("The map visualization is currently unavailable. However, you can still use the chatbot feature.")
    
    # Show only the chatbot if map data is missing
    st.markdown("### Chatbot")
    messages = st.container(height=300)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with messages.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input("Ask about potholes or road conditions"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with messages.chat_message("user"):
            st.write(prompt)

        # Get response from Groq AI
        response = get_groq_response(prompt)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        with messages.chat_message("assistant"):
            st.write(response)

