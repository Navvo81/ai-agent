import os
import sys
from dotenv import load_dotenv
from google import genai
from google.genai import types
from config import SYSTEM_PROMPT
from schema import schema_get_files_info, schema_get_files_content, schema_run_python_file, schema_write_file
from functions import get_file_content, get_files_info, run_python_file, write_file

functions = {
    "get_file_content": get_file_content.get_file_content,
    "get_files_info":get_files_info.get_files_info,
    "run_python_file":run_python_file.run_python_file,
    "write_file":write_file.write_file
}

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

    iters = 0
    while True:
        iters += 1
        if iters > 20:
            print(f"Maximum iterations reached.")
            sys.exit(1)

        try:
            final_response = generate_content(client, messages, args)
            if final_response:
                print("Final response:")
                print(final_response)
                break
        except Exception as e:
            print(f"Error in generate_content: {e}")

def call_function(function_call_part, verbose=False):
    function_name = function_call_part.name
    if verbose:
        print(f"Calling function: {function_call_part.name}({function_call_part.args})")
    else:
        print(f" - Calling function: {function_call_part.name}")
    function_args = function_call_part.args.copy()
    function_args['working_directory']='./calculator'
    if function_call_part.name not in functions:
        return types.Content(
    role="tool",
    parts=[
        types.Part.from_function_response(
            name=function_name,
            response={"error": f"Unknown function: {function_name}"},
        )
    ],
    )
    else:
        function_result = functions[function_call_part.name](**function_args)
        return types.Content(
        role="tool",
        parts=[
            types.Part.from_function_response(
                name=function_name,
                response={"result": function_result},
            )
        ],
        )
    

def generate_content(client, messages, args):

    available_functions = types.Tool(
    function_declarations=[
        schema_get_files_info,
        schema_get_files_content,
        schema_run_python_file,
        schema_write_file
    ]
    )
    response = client.models.generate_content(
        model='gemini-2.0-flash-001',
        contents=messages,
        config=types.GenerateContentConfig(
            tools=[available_functions],
            system_instruction=SYSTEM_PROMPT),
        )
    verbose_flag = (args == '--verbose')

    if verbose_flag:
        print(f"Prompt tokens: {response.usage_metadata.prompt_token_count}\nResponse tokens: {response.usage_metadata.candidates_token_count}")
        print

    if response.candidates:
        for candidate in response.candidates:
            function_call_content = candidate.content
            messages.append(function_call_content)

    if not response.function_calls:
        return response.text

    function_responses = []
    for function_call_part in response.function_calls:
        function_call_result = call_function(function_call_part, verbose_flag)
        if (
            not function_call_result.parts
            or not function_call_result.parts[0].function_response
        ):
            raise Exception("empty function call result")
        if verbose_flag:
            print(f"-> {function_call_result.parts[0].function_response.response}")
        function_responses.append(function_call_result.parts[0])

    if not function_responses:
        raise Exception("no function responses generated, exiting.")

    messages.append(types.Content(role="tool", parts=function_responses))

    '''

    if response.function_calls:
        for function_call_part in response.function_calls:
            function_call_result = call_function(function_call_part, verbose_flag)
            if function_call_result.parts and hasattr(function_call_result.parts[0], 'function_response') \
            and hasattr(function_call_result.parts[0].function_response, 'response'):
                if verbose_flag:
                    print(f"-> {function_call_result.parts[0].function_response.response}")
            else:
                raise Exception("Error:  no function called")
    else:
        print(response.text)
    '''
    
if __name__ == "__main__":
    main()