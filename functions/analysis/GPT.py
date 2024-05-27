import requests
import os


def query_chatgpt(prompt):
    api_key = os.environ['GPT_KEY']

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    pre_prompt = (
        "Please give me a detailed answer with evidentical explanation. Analyze a given information about my Graph which represents "
        "a bigdata or complex system architecture where I'm testing if it's able to handle"
        "the high load. Basically I need a recommendation what to do. Provide details about risk-group nodes and "
        "issues you are seeing, give exact nodes you've identified as questionable and exact parts."
        "Talk not with a Graph Theory Terminology, but with solutions architecture. Provide as much details as possible.")

    prompt = pre_prompt + prompt

    data = {
        "model": "gpt-3.5-turbo",  # Or the appropriate model you have access to
        "messages": [{"role": "user", "content": prompt}],
    }

    response = requests.post(url, headers=headers, json=data)

    response = response.json()

    if response.get("error"):
        print("Error:", response["error"])
    else:
        messages = response.get("choices", [{}])[0].get("message", {})
        response = messages['content']

        return response
