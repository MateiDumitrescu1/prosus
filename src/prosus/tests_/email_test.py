import openai
client = openai.OpenAI(
    api_key="sk-UFVXQZmZA4oRQQjJERojmA",
    base_url="https://pd67dqn1bd.execute-api.eu-west-1.amazonaws.com" 
)

response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages = [
        {
            "role": "user",
            "content": "this is a test request, write a short poem"
        }
    ]
)

print(response)