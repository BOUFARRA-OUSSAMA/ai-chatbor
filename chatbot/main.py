import json
import os
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

template = """
Answer the questions below.

Here is the conversation history = {context}

Question: {question}

Answer:
"""
model = OllamaLLM(model="llama3.2:1b")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def save_conversation(context, filename="conversation_history.json"):
    with open(filename, "w") as f:
        json.dump(context, f, indent=2)
    print(f"Conversation saved to {filename}")

def load_conversation(filename="conversation_history.json"):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    return []

def handle_conversation():
    # Load previous conversation or start fresh
    context = load_conversation()
    print("Welcome to the conversation bot! Type 'exit' to end the conversation.")
    print("Type 'new' to start a fresh conversation.")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() == 'exit':
            save_conversation(context)
            print("Conversation saved. Ending conversation.")
            break
            
        if user_input.lower() == 'new':
            save_conversation(context)  # Save the old conversation first
            context = []
            print("Started a fresh conversation.")
            continue
            
        # Format context for the model (convert structured data to string)
        context_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in context])
        
        result = chain.invoke({
            "context": context_text,
            "question": user_input
        })
        
        # Save both messages in structured format
        context.append({"role": "user", "content": user_input})
        context.append({"role": "bot", "content": result})
        
        print(f"Bot: {result}")

if __name__ == "__main__":
    handle_conversation()