# San Antonio Pothole Map & Chatbot Documentation

This document provides a detailed overview of the implementation approaches for the chatbot, data visualization, and map integration in the `integrated_prototype/integrated.py` application.

## 1. Setup and Running the Application

To run the Streamlit application, ensure you have all necessary dependencies installed (e.g., `streamlit`, `folium`, `geopandas`, `pandas`, `requests`, `matplotlib`, `seaborn`, `numpy`). You also need the `Data` folder containing `311_Pothole_Cases_18_24.csv`, `COSA_Pavement.csv`, and `COSA_pavement_311.csv` in the same directory as `integrated_prototype/integrated.py`.

From your terminal, navigate to the directory containing `integrated_prototype/integrated.py` and run:

```bash
streamlit run integrated_prototype/integrated.py
```

## 2. Chatbot Enhancements (`get_groq_response`)

The `get_groq_response` function acts as the central dispatch for user queries. It analyzes the user's prompt and, based on keywords, directs the request to a specific analytical function.

### Implementation Approaches:

*   **Keyword-Based Dispatch:** The function checks for specific keywords (e.g., "predict new potholes", "worst potholes") in the user's prompt to call relevant analytical functions.
*   **Dynamic Visualization Data:** Analytical functions (like `get_pothole_formation_prediction`, `get_worst_pothole_streets`, `get_top_complaint_locations`) are designed to return:
    *   `response_text`: The textual answer for the chatbot.
    *   `plot_object`: A Matplotlib figure if a chart is generated.
    *   `highlight_data_df`: A Pandas DataFrame containing `Latitude`, `Longitude`, `color`, and `marker_radius` for map visualization.

## 3. Data Loading and Preprocessing (`load_pavement_data`)

The `load_pavement_data` function is crucial for preparing the `pavement_latlon_df` which includes geographical coordinates.

### Implementation Approaches:

*   **Latitude and Longitude Extraction:** The `Latitude` and `Longitude` columns are extracted from the `GoogleMapView` URL column using a regular expression.
*   **Longitude Correction:** A critical fix was implemented to ensure longitude values for San Antonio (Western Hemisphere) are correctly represented as negative. The regex was adjusted to capture the direction ('W' or 'E') alongside the numeric longitude, and the code now explicitly negates the longitude if the direction is 'W'. This resolves issues where the map was displaying locations in the Eastern Hemisphere due to positive longitude values.
*   **NaN Handling:** Rows with missing `MSAG_Name`, `Latitude`, or `Longitude` are dropped to ensure data quality for mapping.

## 4. Map Visualization (`add_pothole_markers`)

The `add_pothole_markers` function is responsible for adding interactive circle markers to the Folium map.

### Implementation Approaches:

*   **Dynamic Marker Properties:** The function now accepts `color_column` and `marker_radius` parameters, allowing different types of insights to be visualized with distinct colors and sizes.
*   **`folium.CircleMarker`:** Instead of `folium.Marker`, `folium.CircleMarker` is used to draw circles, which are more suitable for representing areas or points with a certain impact radius.
*   **Feature Groups:** Markers are added to a `folium.FeatureGroup`, which allows for better organization and control over layers on the map.

## 5. Map Integration and Rendering (`integrated.py` main map section)

The main map rendering logic within the `col1` section handles the display and updates of the Folium map.

### Implementation Approaches:

*   **Dynamic Map Initialization:** The `folium.Map` object and `folium.FeatureGroup` are created dynamically within the `map_placeholder.container()` on each run. This approach helps in avoiding the "Map container is already initialized" error that can occur with `streamlit-folium` when re-rendering.
*   **Dynamic `st_folium` Key:** A dynamic `key` is passed to `st_folium`. This `key` is generated based on the length of chat messages and a hash of the `vis_data` (excluding the `plot_object`), which forces Streamlit to re-render the map when new visualization data is available, ensuring the map updates reflect the latest chatbot response.
*   **JSON Serialization Handling:**
    *   The `plot_object` (Matplotlib Figure) is explicitly excluded from the `vis_data` when generating the hash for the `st_folium` key to prevent `TypeError: Object of type Figure is not JSON serializable`.
    *   A helper function `_convert_dataframe_numerics_to_native_types` was introduced to convert `numpy.int64` and `numpy.float64` types in `highlight_data_df` to native Python `int` or `float`. This addresses `TypeError: Object of type int64 is not JSON serializable` during JSON serialization of the highlight data.
*   **Helper Function Placement:** The `_convert_dataframe_numerics_to_native_types` function was moved to directly after initial imports to ensure it's defined before any calls to it, resolving `NameError` issues.

## 6. Expected Output

When you run the Streamlit application and interact with the chatbot, you should observe the following:

*   **Textual Responses:** The chatbot will provide textual answers to your queries in the right-hand `Chatbot` column.
*   **Plots:** For queries like "worst potholes" or "predict new potholes," a bar chart will appear below the chatbot's textual response.
*   **Map Visualizations:**
    *   **Default Map:** The map will initially center on San Antonio.
    *   **Dynamic Circles:** When you ask questions that trigger map visualizations (e.g., "predict new potholes"), dark blue circles will appear on the map, highlighting specific locations.

## 7. Example Queries

You can use the following queries in the chatbot:

*   "predict new potholes"
*   "worst potholes"
*   "top complaint locations"
*   "What is the pavement condition for MAIN ST?"
*   "How many potholes this month?"
*   "What is the seasonal impact on potholes?" 