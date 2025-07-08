# Pothole Map with AI Chatbot

This project combines a pothole mapping visualization with an AI-powered chatbot using Groq's LLaMA 4 model. The application allows users to view pothole locations on a map and interact with an AI assistant to get information about road conditions.

## Features

- Interactive map visualization of pothole locations
- AI-powered chatbot using Groq's LLaMA 4 model
- Real-time chat history
- Responsive layout with map and chat interface

## Requirements

- Python 3.x
- Streamlit
- Folium
- GeoPandas
- Pandas
- Requests

## Installation

1. Clone the repository:
```bash
git clone [your-repository-url]
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Set up your Groq API key in the code or environment variables.

## Usage

Run the Streamlit app:
```bash
streamlit run integrated_prototype/integrated.py
```

The application will be available at `http://localhost:8501`

## Configuration

- The Groq API key is configured in the code
- Map center coordinates and zoom level can be adjusted in the code
- The number of displayed pothole markers can be modified

## License

[Your chosen license]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
