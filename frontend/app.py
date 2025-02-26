from langchain_ollama.chat_models import ChatOllama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Initialize both LLMs
llm1 = ChatOllama(model="llama3.2", base_url="http://backend:11434")
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

# Set up the sequential chain
def process_query(inputs):
    # Step 1: Translate from Arabic to English
    english_query = translate_to_english_prompt.invoke({"input": inputs["input"]}) | llm1 | StrOutputParser()
    
    # Step 2: Generate response in English
    english_response = english_response_prompt.invoke({
        "english_query": english_query,
        "chat_history": inputs.get("chat_history", [])
    }) | llm2 | StrOutputParser()
    
    # Step 3: Translate from English to Arabic
    return translate_to_arabic_prompt.invoke({"english_response": english_response}) | llm1

# Create the chain with history
history = StreamlitChatMessageHistory()

chain_with_history = RunnableWithMessageHistory(
    RunnablePassthrough.assign(output=process_query),
    lambda session_id: history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

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
    
    # Process and stream response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        # Process the chain and stream results
        for chunk in chain_with_history.stream(
            {"input": question}, 
            config={"configurable": {"session_id": "any"}}
        ):
            if chunk.get("output"):
                full_response += chunk["output"]
                response_placeholder.markdown(
                    f'<div style="text-align: right; direction: rtl;">{full_response}</div>',
                    unsafe_allow_html=True
                )