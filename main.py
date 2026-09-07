import dotenv
import os
from openai import OpenAI

dotenv.load_dotenv()

llm = OpenAI(
    api_key=os.getenv('GROQ_APIKEY'),
    base_url='https://api.groq.com/openai/v1'
)

messages = [
    {'role': 'system', 'content': 'You are Pouya chat bot assistant.'},
]

total_tokens = 0


def bot(message):
    global total_tokens

    messages.append({'role': 'user', 'content': message})
    response = llm.chat.completions.create(
        model='qwen/qwen3.8-27b',
        messages=messages,
        temperature=0.7,
        max_tokens=100,
        # stream=True
    )

    response_text = response.choices[0].message.content
    messages.append({'role': 'assistant', 'content': response_text})

    total_tokens = total_tokens + response.usage.total_tokens
    return response_text


def chat():
    while True:
        user_input = input("Enter your message (type x to terminate) : ").strip().lower()

        if user_input == 'x':
            print("Goodbye!")
            break

        print(bot(user_input))


def main():
    chat()
    print(f'total tokens: {total_tokens}')


if __name__ == '__main__':
    main()
