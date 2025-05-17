# _________________________________________________________________________________________________
# ______________________________________ deepseek (using API) _____________________________________

from openai import OpenAI, APIError, AuthenticationError
"""
ak = "sk-c081fe785d5a4acd8fd4e7a73d227222"
client = OpenAI(api_key=ak, base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Descibe the landscape of this location: 38.304052075548185 40.10125499999177"},
    ],
    stream=False,
    temperature=0.5,
    max_tokens=150
)
"""
#print(response.choices[0].message.content)

# _______________________________________________________________________________________________
# ______________________________________ ollama _________________________________________________

from ollama import chat, generate
from ollama import ChatResponse

response: ChatResponse = chat(
    model='deepseek-r1', # other models deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B, llama3.2 
    messages=[
    {'role': 'user', 
    'content': 'Descibe the landscape of this location: 38.304052075548185 40.10125499999177',
  },
])
print(response['message']['content'])
# # # or access fields directly from the response object
# print(response.message.content)

