import os
import sys
from dotenv import load_dotenv
from google import genai
from google.genai import types

def main():

    load_dotenv()


    args = sys.argv[1:]

    if not args:
        print("No query supplied")
        print("\nUsage: python3 main.py 'your query here'")
        sys.exit(1)
    user_prompt = " ".join(args)

    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    messages = [
        types.Content(role="user", parts=[types.Part(text=user_prompt)]),
                ]

    generate_content(client, messages, sys.argv[-1])

def generate_content(client, messages, args):
    response = client.models.generate_content(model='gemini-2.0-flash-001',contents=messages,)
    if args == '--verbose':
        print(f"User prompt: {messages}")
        print(f"\nResponse: {response.text}")
        print(f"Prompt tokens: {response.usage_metadata.prompt_token_count}\nResponse tokens: {response.usage_metadata.candidates_token_count}")
    else:
        print(response.text)
    
if __name__ == "__main__":
    main()