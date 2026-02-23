from openai import OpenAI

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="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",
)

completion = client.chat.completions.create(
  model="openai/gpt-oss-120b",
  messages=[
    {
      "role": "user",
      "content": "What is the meaning of life?"
    }
  ]
)

print(completion.choices[0].message.content)
