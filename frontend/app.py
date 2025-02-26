from langchain_ollama.chat_models import ChatOllama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.output_parsers import StrOutputParser

# Initialize both LLMs
llm1 = ChatOllama(model="llama3.2:3b", base_url="http://backend:11434")
llm2 = ChatOllama(model="deepseek-r1:8b", base_url="http://backend:11434")

# Create translation prompt for Arabic to English
translate_to_english_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a professional translator. Translate the following Arabic text to English accurately. "
        "Only provide the translated text without any additional comments or explanations."
    ),
    ("human", "{input}")
])

# Create English response generation prompt
english_response_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an AI assistant answering questions in English. Provide informative and helpful responses."
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{english_query}")
])

# Create translation prompt for English to Arabic
translate_to_arabic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a professional translator. Translate the following English text to Arabic accurately. "
        "Make the translation fluent and natural. Include appropriate emojis to enhance the message."
    ),
    ("human", "{english_response}")
])

# Create the translation chains
ar_to_en_chain = translate_to_english_prompt | llm1 | StrOutputParser()
en_to_ar_chain = translate_to_arabic_prompt | llm1 | StrOutputParser()

# Set up the response chain
def generate_english_response(english_query, history):
    prompt_value = english_response_prompt.invoke({
        "english_query": english_query,
        "chat_history": history
    })
    response = llm2.invoke(prompt_value)
    return response.content

# Define the complete chain with history
history = StreamlitChatMessageHistory()

def sequential_chain(inputs, config):
    # Extract chat history from inputs
    chat_history = inputs.get("chat_history", [])
    
    # Step 1: Translate Arabic to English
    english_query = ar_to_en_chain.invoke({"input": inputs["input"]})
    
    # Step 2: Generate English response
    english_response = generate_english_response(english_query, chat_history)
    
    # Step 3: Translate English to Arabic
    arabic_response = en_to_ar_chain.invoke({"english_response": english_response})
    
    # Return the result
    return {"output": arabic_response}

# Set up Streamlit UI with RTL support
st.markdown(
    """
    <style>
    body {
        direction: rtl;
        text-align: right;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("نظام الذكاء الاصطناعي لكلية القيادة والأركان")

# Initialize session state for messages if not already done
if "langchain_messages" not in st.session_state:
    st.session_state["langchain_messages"] = []

# Display chat history
for message in st.session_state["langchain_messages"]:
    role = "user" if message.type == "human" else "assistant"
    with st.chat_message(role):
        st.markdown(
            f'<div style="text-align: right; direction: rtl;">{message.content}</div>',
            unsafe_allow_html=True
        )

# Handle user input
question = st.chat_input("السؤال الخاص بك")
if question:
    # Display user message
    with st.chat_message("user"):
        st.markdown(
            f'<div style="text-align: right; direction: rtl;">{question}</div>',
            unsafe_allow_html=True
        )
    
    # Add message to history
    history.add_user_message(question)
    
    # Process and stream response
    with st.chat_message("assistant"):
        response_container = st.empty()
        
        # Process without streaming for simplicity (to fix the error)
        response = sequential_chain(
            {"input": question, "chat_history": history.messages}, 
            {"configurable": {"session_id": "any"}}
        )
        
        arabic_response = response["output"]
        response_container.markdown(
            f'<div style="text-align: right; direction: rtl;">{arabic_response}</div>',
            unsafe_allow_html=True
        )
        
        # Add assistant response to history
        history.add_ai_message(arabic_response)