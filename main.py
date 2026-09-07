import dotenv
import os
from openai import OpenAI

dotenv.load_dotenv()


class ChatBot:
    def __init__(self):
        self.llm = OpenAI(
            api_key=os.getenv('GROQ_APIKEY'),
            base_url='https://api.groq.com/openai/v1'
        )

        self.messages = [
            {'role': 'system', 'content': 'You are Pouya chat bot assistant.'},
        ]

        self.total_tokens = 0

    def bot(self, message):

        self.messages.append({'role': 'user', 'content': message})
        response = self.llm.chat.completions.create(
            model='qwen/qwen3.8-27b',
            messages=self.messages,
            temperature=0.7,
            max_tokens=100,
            # stream=True
        )

        response_text = response.choices[0].message.content
        self.messages.append({'role': 'assistant', 'content': response_text})

        self.total_tokens = self.total_tokens + response.usage.total_tokens
        return response_text

    def chat(self):
        while True:
            user_input = input("Enter your message (type x to terminate) : ").strip().lower()

            if user_input == 'x':
                print("Goodbye!")
                break

            print(self.bot(user_input))

    def report(self):
        for message in self.messages:
            print(message)

        print(f'total tokens: {self.total_tokens}')


def main():
    bot = ChatBot()
    bot.chat()
    bot.report()


if __name__ == '__main__':
    main()
