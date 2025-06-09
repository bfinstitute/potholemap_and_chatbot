import streamlit as st
from openai import OpenAI
import folium
import geopandas as gpd
import pandas as pd
from streamlit_folium import st_folium

# ---------- App Configuration ----------
st.set_page_config(layout="wide")
st.title('🗺️ Integrated Chat with Map')


# ---------- Load and Plot Map ----------
my_subset = gpd.read_file("potholes_data.gpkg", layer="potholes")  # load pre-cleaned geopackage

small_df = my_subset.head(10)

def plot_from_df(df, folium_map):
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
    if prompt := st.chat_input("Say something"):
        messages.chat_message("user").write(prompt)
        messages.chat_message("assistant").write(f"Echo: {prompt}")