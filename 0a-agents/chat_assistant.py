
import json

from IPython.display import display, HTML
import markdown
from google.genai import types

class Tools:
    def __init__(self):
        self.tools = {}
        self.functions = {}

    def add_tool(self, function, description):
        
        self.tools[function.__name__] = description
        self.functions[function.__name__] = function
    
    def get_tools(self):
        return list(self.tools.values())

    def function_call(self, tool_call_response):
        function_name = tool_call_response.name
        arguments = tool_call_response.args

        f = self.functions[function_name]
        result = f(**arguments)

        function_response_part = types.Part.from_function_response(
            name=function_name,
            response={"result": result},
        )

        return function_response_part


def shorten(text, max_length=50):
    if len(text) <= max_length:
        return text

    return text[:max_length - 3] + "..."


class ChatInterface:
    def input(self):
        question = input("You:")
        return question
    
    def display(self, message):
        print(message)

    def display_function_call(self, entry, result):
        call_html = f"""
            <details>
            <summary>Function call: <tt>{entry.name}({entry.args})</tt></summary>
            <div>
                <b>Call</b>
                <pre>{entry}</pre>
            </div>
            <div>
                <b>Output</b>
                <pre>{result}</pre>
            </div>
            
            </details>
        """
        display(HTML(call_html))

    def display_response(self, entry):
        response_html = markdown.markdown(entry)
        html = f"""
            <div>
                <div><b>Assistant:</b></div>
                <div>{response_html}</div>
            </div>
        """
        display(HTML(html))



class ChatAssistant:
    def __init__(self, tools, developer_prompt, chat_interface, client):
        self.developer_prompt = developer_prompt
        self.chat_interface = chat_interface
        self.client = client
        self.tools = tools
        declared_tools = []
        for tool in self.tools.get_tools():
            declared_tools.append(types.Tool(function_declarations=[tool]))
        self.config = types.GenerateContentConfig(tools=declared_tools)
    
    def gpt(self, chat_messages):
        return self.client.models.generate_content(
                model="gemini-2.5-flash",
                config=self.config,
                contents=chat_messages,
            )

    def run(self):
        chat_messages = []

        # Chat loop
        while True:
            question = self.chat_interface.input()
            if question.strip().lower() == 'stop':  
                self.chat_interface.display("Chat ended.")
                break

            chat_messages.append(self.developer_prompt.format(question=question))

            while True:  # inner request loop
                response = self.gpt(chat_messages)

                has_messages = False

                for entry in response.candidates:
                    for part in entry.content.parts:
                        chat_messages.append(part)
                    
                        if part.function_call: 
                            result = self.tools.function_call(part.function_call)
                            chat_messages.append(result)
                            self.chat_interface.display_function_call(part.function_call, result)
                        elif part.text == '\n':
                            print(part.text) 
                            has_messages = False
                        elif part.text:
                            self.chat_interface.display_response(part.text)
                            has_messages = True

                if has_messages:
                    break

                
    

