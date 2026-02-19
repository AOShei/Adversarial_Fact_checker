# Agentic Adversarial Truth Analysis Framework

A multi-agent AI system designed to reduce hallucination and confirmation bias when analyzing text-based reports.

## Current Behavior (Important)

- Default provider in the UI is **Microsoft Azure**.
- Azure calls are routed through `app/llm.py`:
    - tries **Responses API** first (required for many GPT-5 deployments),
    - falls back to **Chat Completions** for older-compatible deployments.
- Claim extraction rewrites claims to be standalone/searchable and now explicitly expands in-document abbreviations/acronyms where possible (e.g., include long-form + shorthand context).
- Web search is intentionally bounded for responsiveness (timeouts + capped result volume) to avoid UI hangs during parallel claim analysis.

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
    AZURE_OPENAI_DEPLOYMENT_NAME=...
    AZURE_OPENAI_API_VERSION=...
    ```

    Azure notes:
    - Use the resource root endpoint (example: `https://<resource>.cognitiveservices.azure.com/`).
    - Do not paste full operation URLs like `/openai/responses?...` into `AZURE_OPENAI_ENDPOINT`.
    - GPT-5 deployments require `AZURE_OPENAI_API_VERSION` `2025-03-01-preview` or later.

3.  **Run the Application:**
    ```bash
    streamlit run app/main.py
    ```

## Azure Quick Troubleshooting

- **400: Responses API enabled only for 2025-03-01-preview+**
    - Set `AZURE_OPENAI_API_VERSION=2025-03-01-preview` or newer (recommended: `2025-04-01-preview`).
- **Connection error**
    - Ensure endpoint is the resource root (for example `https://<resource>.cognitiveservices.azure.com/`) and not a full operation URL.
- **404 / deployment not found**
    - Verify `AZURE_OPENAI_DEPLOYMENT_NAME` exactly matches the deployed model name in Azure.
- **Auth failures (401/403)**
    - Confirm the API key belongs to the same resource as the endpoint.

## Project Structure

-   `app/`: Main application code.
    -   `main.py`: Streamlit entry point.
    -   `agents/`: Individual agent logic.
    -   `llm.py`: LLM provider abstraction.
    -   `database.py`: SQLite history management.
    -   `models.py`: Data models.
