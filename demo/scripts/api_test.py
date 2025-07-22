from openai import OpenAI

client = OpenAI(api_key="Empty", base_url="http://localhost:8001/v1")

message = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the meaning of life?"}
]
response = client.chat.completions.create(
    model="qwen-2",
    messages=message,
    temperature=0.2,
    max_tokens=5000
)

print(response.choices[0].message.content)