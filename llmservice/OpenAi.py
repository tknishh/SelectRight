import openai

class OpenAi:
    def __init__(self, apiKey, options={}, socket=None):
        self.apiKey = apiKey
        self.options = options
        self.socket = socket
        openai.api_key = self.apiKey

    def ask(self, context, question, functions=[]):
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=context,
                # functions=functions,
                max_tokens=150,
                n=1,
                stop=None,
                temperature=0.7
            )
            return {
                'msg': response.choices[0].message['content'],
                'raw': response
            }
        except Exception as e:
            print(f"Error in OpenAi.ask: {e}")
            return None