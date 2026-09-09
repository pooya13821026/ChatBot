import time

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

        self.iteration = 0

    def summerize_message(self):
        if self.iteration % 5 != 0:
            return

        system_message = [m for m in self.messages if m['role'] == 'system']
        other_message = [m for m in self.messages if m['role'] != 'system']

        first_messages, last_messages = (other_message[0:5], other_message[5:])

        response = self.llm.chat.completions.create(
            model='qwen/qwen3.8-27b',
            messages=[{'role': 'system', 'content': 'summerize these messages'}, *first_messages],
            temperature=0.7
        )

        summerized_message = {'role': 'system',
                              'content': f'summery of 5 messages :{response.choices[0].message.content}'
                              }

        self.messages = [*system_message, summerized_message, *last_messages]

    def trim_message(self, max_count=5):
        if len(self.messages) < max_count:
            return

        system_message = [m for m in self.messages if m['role'] == 'system']
        other_message = [m for m in self.messages if m['role'] != 'system']

        trimed_messages = other_message[-max_count:]

        self.messages = [*system_message, *trimed_messages]

    def bot(self, message):

        self.messages.append({'role': 'user', 'content': message})
        response = self.llm.chat.completions.create(
            model='qwen/qwen3.8-27b',
            messages=self.messages,
            temperature=0.7,
            # max_tokens=1000,
            stream=True
        )

        response_text = ''
        for chunk in response:
            if chunk.choices:
                delta = chunk.choices[0].delta.content

                if delta:
                    print(delta, end='', flush=True)
                    response_text += delta

            if chunk.usage:
                self.total_tokens = self.total_tokens + chunk.usage.total_tokens
                self.input_tokens = self.input_tokens + chunk.usage.prompt_tokens
                self.output_tokens = self.output_tokens + chunk.usage.completion_tokens

        print()
        self.messages.append({'role': 'assistant', 'content': response_text})

        self.summerize_message()

    def chat(self):
        encoding = AutoTokenizer.from_pretrained('Qwen/Qwen3.8-27B')
        while True:
            self.iteration += 1
            user_input = input("Enter your message (type x to terminate) : ").strip().lower()

            if user_input == 'x':
                print("Goodbye!")
                break

            tokens_count = len(encoding.encode(user_input))
            if tokens_count > 100:
                print('your prompt exceeded limit, please make shorter')
                continue

            if tokens_count > 10000:
                print('unfortunately you exceeded the total limit for today.')

            result = self.bot(user_input)
            print(result)

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
