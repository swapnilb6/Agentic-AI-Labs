from openai import OpenAI
client = OpenAI()

 

response = client.responses.create(
  model="gpt-4o",
  input=[
    {
      "role": "system",
      "content": [
        {
          "type": "input_text",
          "text": "You will be provided with statements, and your task is to convert them to standard English."
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "input_text",
          "text": "She no went to the market."
        }
      ]
    }
  ],
  temperature=0.1,
  max_output_tokens=256
)

 

print(response.output_text)

 