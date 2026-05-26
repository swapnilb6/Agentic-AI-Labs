# import openai
 
from openai import OpenAI
client = OpenAI()
 
 
#client = openai.OpenAI(api_key="")
 
prompt = """Classify sentiment as negative, positive, or neutral:
1. Amazingly crafted and good!
2. This thing broke after the second use.
3. Yes it looks good, but they don't feel good.
4. Can't wait to show them off to my partner!"""
 
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=100
)
 
print(response.choices[0].message.content)