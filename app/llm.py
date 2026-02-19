from google import genai
from openai import AzureOpenAI
from typing import Dict, Any

def call_llm(prompt: str, provider: str, config: Dict[str, Any]) -> str:
    """Routes the prompt to the selected API provider."""
    
    if provider == "Google Gemini":
        api_key = config.get("gemini_key")
        if not api_key:
            return "Error: Gemini API Key missing."
        try:
            client = genai.Client(api_key=api_key)
            model_name = config.get("gemini_model", "gemini-2.5-flash")
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            return response.text
        except Exception as e:
            return f"Gemini Error: {str(e)}"

    elif provider == "Microsoft Azure":
        required = ["azure_endpoint", "azure_key", "azure_version", "azure_deployment"]
        if not all(config.get(k) for k in required):
            return "Error: Missing Azure configuration details."
        try:
            client = AzureOpenAI(
                azure_endpoint=config["azure_endpoint"],
                api_key=config["azure_key"],
                api_version=config["azure_version"]
            )
            response = client.chat.completions.create(
                model=config["azure_deployment"],
                messages=[
                    {"role": "system", "content": "You are a precise analytical agent. Return only the requested information."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Azure Error: {str(e)}"
    
    return "Error: Invalid Provider"
