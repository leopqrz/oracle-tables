import openai
import tiktoken
from currency_converter import CurrencyConverter
c = CurrencyConverter()
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

''' 
To set a OPENAI_API_KEY follow the instructions:
1 - Open the link: https://platform.openai.com/account/api-keys
2 - Create a new secret key
3 - Copy the secret key
4 - Open the terminal (e.g. CMD, Iterm2)
5 - Type: setx OPEN_API_KEY "paste your secret key here" and enter
6 - Open a new terminal and type: echo %OPENAI_API_KEY"
7 - You should see the secret key after the previous step
8 - You may need to restart your OS
'''

openai.api_key = os.getenv('OPENAI_API_KEY')
model = "gpt-3.5-turbo"


def get_completion(prompt, model=model):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,  # Degree of randomness of the model's output
    )
    return response.choices[0].message["content"]


# Check Tokens

'''
Encoding name	        OpenAI models

cl100k_base	            gpt-4, gpt-3.5-turbo, text-embedding-ada-002
p50k_base	            Codex models, text-davinci-002, text-davinci-003
r50k_base (or gpt2)	    GPT-3 models like davinci
'''
# Load an encoding
enc = tiktoken.get_encoding("cl100k_base")

# To get the tokeniser corresponding to a specific model in the OpenAI API:
enc = tiktoken.encoding_for_model(model)


def num_tokens_from_string(
        string: str,
        encoding_name: str = "cl100k_base",
        price=0.002/1000
        ) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    msg = None
    if num_tokens >= 4097:
        msg = " (Maximum #Tokens: 4097)"
    print(f"Number of Tokens: {num_tokens}{msg}\
          \nPrice: \n\tUSD$ {round(num_tokens*price,2)}\
          \n\tCAD$ {round(c.convert(num_tokens*price, 'USD', 'CAD'),2)}")
    # return (num_tokens, num_tokens*price, c.convert(num_tokens*price, 'USD', 'CAD'))