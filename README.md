# ExoBot 🚀

ExoBot is an interactive AI-powered chatbot designed to answer questions about exoplanets. Built for the NASA Space Apps Cairo Hackathon, it leverages generative AI (Google Gemini via LangChain) and a CSV knowledge base to provide informative, friendly, and concise responses about exoplanet discovery and features. The project includes data scraping, feature engineering, conversational memory, and a Streamlit web interface.

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Data Sources](#data-sources)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Development Process](#development-process)
- [License](#license)

---

## Features

- Conversational chatbot interface about exoplanets
- Uses Google Gemini (via `langchain-google-genai`) for AI responses
- Retrieves facts from a local CSV file (`exoplanets_cleaned.csv`)
- Supports follow-up questions and remembers chat history
- Multilingual and can translate between languages
- Customizable UI with a background image
- Data scraping and feature extraction from NASA Exoplanet Archive
- Feature and description mapping for user-friendly answers

---

## Project Structure

```
memory.txt
README.md
data/
    exoplanets.csv
    exoplanet_archive_columns.csv
    Features.txt
    more_features.txt
Explanets-space-bot/
    background.jpg
    exoplanets_cleaned.csv
    README.md
    requirements.txt
    streamlit_app.py
    exoplanets_vectorstore/
        index.faiss
        index.pkl
notebooks/
    scraping.ipynb
    test.ipynb
    exoplanets_vectorstore/
        index.faiss
        index.pkl
```

- `Explanets-space-bot/streamlit_app.py` — Main Streamlit application code
- `Explanets-space-bot/requirements.txt` — Python dependencies
- `data/exoplanets.csv` — Raw exoplanet data
- `data/exoplanet_archive_columns.csv` — Feature descriptions scraped from NASA
- `data/Features.txt`, `data/more_features.txt` — Feature name mappings and extra descriptions
- `memory.txt` — Stores chat history for context
- `notebooks/test.ipynb` — Data exploration, feature engineering, and prototyping
- `notebooks/scraping.ipynb` — Scripts for scraping feature descriptions
- `background.jpg` — Background image for the UI

---

## Data Sources

- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)
- [API_PS_columns documentation](https://exoplanetarchive.ipac.caltech.edu/docs/API_PS_columns.html) (for feature descriptions)

---

## Installation

1. **Clone the repository and submodules:**

    ```powershell
    git clone --recurse-submodules https://github.com/mhmdkardosha/Exoplanets-space-bot
    cd Exoplanets-space-bot
    ```

2. **Install dependencies:**

    ```powershell
    pip install -r requirements.txt
    ```

3. **Add your `.env` file** with the necessary API keys (e.g., for Google Generative AI).

4. **Ensure data files** (`exoplanets_cleaned.csv`, `background.jpg`) are present in the project directory.

---

## Usage

Run the Streamlit app:

```powershell
streamlit run Explanets-space-bot/streamlit_app.py
```

Open the provided local URL in your browser to interact with ExoBot.

---

## How It Works

- **Data Preparation:**  
  - Exoplanet data is downloaded from NASA and cleaned.
  - Feature descriptions are scraped and mapped for user-friendly answers.

- **Conversational AI:**  
  - Uses Google Gemini via LangChain for natural language understanding and generation.
  - Maintains chat history in `memory.txt` for context-aware responses.

- **Retrieval QA:**  
  - Loads exoplanet data and creates vector embeddings for retrieval.
  - Answers questions using a retrieval QA chain from the CSV.
  - Falls back to generative AI for general or follow-up questions.

- **User Interface:**  
  - Built with Streamlit for an interactive web experience.
  - Customizable background and wide layout for better usability.

---

## Development Process

1. **Data Acquisition:**  
   Downloaded exoplanet data from NASA and scraped feature descriptions using BeautifulSoup.

2. **Data Cleaning & Feature Engineering:**  
   - Mapped raw feature names to human-readable labels using `Features.txt` and `more_features.txt`.
   - Explored and validated data in Jupyter notebooks.

3. **Conversational AI Prototyping:**  
   - Prototyped prompt engineering and memory handling in `notebooks/test.ipynb`.
   - Implemented chat history for context.

4. **Streamlit App:**  
   - Developed the main app in `Explanets-space-bot/streamlit_app.py`.
   - Integrated retrieval QA and generative fallback.
   - Added UI enhancements and background image support.

5. **Testing & Iteration:**  
   - Tested with various user queries.
   - Improved feature mapping, fallback logic, and response clarity.

---

## License

MIT License

---

#### Made for NASA Space Apps Cairo Hackathon
