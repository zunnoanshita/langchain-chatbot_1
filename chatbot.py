from langchain_community.llms import Ollama
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# Load Ollama model
llm = Ollama(model="gemma:2b")

# Memory for conversation
memory = ConversationBufferMemory()

# Custom prompt template
prompt = PromptTemplate(
    input_variables=["history", "input"],
    template="""
You are ChatGem, a friendly and knowledgeable AI assistant. You love helping users with clear, conversational answers.

Use the conversation history to stay in context. Never say you're just a language model. If you don't know something, take a best guess and say so nicely.

Conversation so far:
{history}
User: {input}
ChatGem:"""
)

# Create conversation chain
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=prompt,
    verbose=True  # optional, shows internals
)

# CLI Chat loop
print("🔮 Chatbot is ready! Type 'exit' to quit.\n")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    response = conversation.predict(input=user_input)
    print(f"Bot: {response}")