import streamlit as st
from langchain_community.llms import Ollama
from langchain import ConversationChain, PromptTemplate
from langchain.memory import ConversationBufferMemory

# Initialize Ollama LLM
llm = Ollama(model="gemma:2b")

# Setup conversation memory
memory = ConversationBufferMemory()

# Setup prompt template
prompt = PromptTemplate(
    input_variables=["history", "input"],
    template="""
You are ChatGem, a helpful and smart AI assistant.

Use the conversation history to answer questions conversationally.

Conversation so far:
{history}
User: {input}
ChatGem:
"""
)

# Setup conversation chain
conversation = ConversationChain(llm=llm, memory=memory, prompt=prompt)

# Streamlit UI
st.title("ChatGem - Your AI Chatbot")

if "chat" not in st.session_state:
    st.session_state.chat = []

# User input box
user_input = st.text_input("You:", key="input")

if user_input:
    response = conversation.predict(input=user_input)
    st.session_state.chat.append({"user": user_input, "bot": response})

# Display chat history
for chat in st.session_state.chat:
    st.markdown(f"**You:** {chat['user']}")
    st.markdown(f"**ChatGem:** {chat['bot']}")
