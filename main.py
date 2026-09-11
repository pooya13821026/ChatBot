import os
import dotenv
from openai import OpenAI
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

        self.tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3.8-27B')

        self.total_tokens = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.iteration = 0

    def summarize_message(self):
        if self.iteration % 5 != 0 or len(self.messages) <= 6:
            return

        system_prompt = self.messages[0]
        conversation = [m for m in self.messages if m['role'] != 'system']

        first_messages = conversation[:5]
        remaining_messages = conversation[5:]

        response = self.llm.chat.completions.create(
            model='qwen/qwen3.8-27b',
            messages=[
                {'role': 'system', 'content': 'Summarize these messages concisely:'},
                *first_messages
            ],
            temperature=0.7
        )

        summary_content = response.choices[0].message.content
        summary_message = {'role': 'system', 'content': f'Summary of previous context: {summary_content}'}

        self.messages = [system_prompt, summary_message, *remaining_messages]

    def bot(self, message):
        self.messages.append({'role': 'user', 'content': message})

        response = self.llm.chat.completions.create(
            model='qwen/qwen3.8-27b',
            messages=self.messages,
            temperature=0.7,
            stream=True
        )

        response_text = ''
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                delta = chunk.choices[0].delta.content
                print(delta, end='', flush=True)
                response_text += delta

            if hasattr(chunk, 'usage') and chunk.usage:
                self.total_tokens += chunk.usage.total_tokens
                self.input_tokens += chunk.usage.prompt_tokens
                self.output_tokens += chunk.usage.completion_tokens

        print()
        self.messages.append({'role': 'assistant', 'content': response_text})
        self.summarize_message()

    def chat(self):
        print("ChatBot initialized. Type 'x' to exit.\n")
        while True:
            user_input = input("User: ").strip()

            if user_input.lower() == 'x':
                print("Goodbye!")
                break

            if not user_input:
                continue

            tokens_count = len(self.tokenizer.encode(user_input))

            if tokens_count > 500:
                print('Your prompt exceeded the limit (max 500 tokens). Please shorten it.')
                continue

            print("Bot: ", end="")
            self.bot(user_input)

    def report(self):
        print("\n--- Usage Report ---")
        print(f'Total tokens: {self.total_tokens}')
        print(f'Input tokens: {self.input_tokens}')
        print(f'Output tokens: {self.output_tokens}')

        cost = (self.input_tokens * 0.05 + self.output_tokens * 0.08) / 1_000_000
        print(f'Estimated Cost: ${cost:.6f}')


def main():
    bot = ChatBot()
    bot.chat()
    bot.report()


if __name__ == '__main__':
    main()
