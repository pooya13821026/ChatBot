import dotenv
import os
from openai import OpenAI
import tiktoken
from transformers import AutoTokenizer

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
        self.input_tokens = 0
        self.output_tokens = 0

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
        self.input_tokens = self.input_tokens + response.usage.prompt_tokens
        self.output_tokens = self.output_tokens + response.usage.completion_tokens

        return response_text

    def chat(self):
        encoding = AutoTokenizer.from_pretrained('Qwen/Qwen3.8-27B')
        while True:
            user_input = input("Enter your message (type x to terminate) : ").strip().lower()

            if user_input == 'x':
                print("Goodbye!")
                break

            tokens_count = len(encoding.encode(user_input))
            print(tokens_count)
            if tokens_count > 10:
                print('your prompt exceeded limit, please make shorter')
                continue

            if tokens_count > 100:
                print('unfortunately you exceeded the total limit for today.')

            print(self.bot(user_input))

    def report(self):
        for message in self.messages:
            print(message)

        print(f'total tokens: {self.total_tokens}')
        print(f'input tokens: {self.input_tokens}')
        print(f'output tokens: {self.output_tokens}')
        print(f'$: {(self.input_tokens * 0.05 + self.output_tokens * 0.08) / 1000000}')


def main():
    bot = ChatBot()
    bot.chat()
    bot.report()


if __name__ == '__main__':
    main()
