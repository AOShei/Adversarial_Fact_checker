# Agentic Adversarial Truth Analysis Framework

A multi-agent AI system designed to reduce hallucination and confirmation bias when analyzing text-based reports.

## Setup

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Environment Variables (MANDATORY):**
    Copy `.env.example` to `.env` and fill in your API keys. The application will not run without these.
    ```bash
    cp .env.example .env
    ```
    
    Edit `.env` to include your provider keys:
    ```
    GEMINI_API_KEY=...
    AZURE_OPENAI_ENDPOINT=...
    AZURE_OPENAI_API_KEY=...
    ```

3.  **Run the Application:**
    ```bash
    streamlit run app/main.py
    ```

## Project Structure

-   `app/`: Main application code.
    -   `main.py`: Streamlit entry point.
    -   `agents/`: Individual agent logic.
    -   `llm.py`: LLM provider abstraction.
    -   `database.py`: SQLite history management.
    -   `models.py`: Data models.
