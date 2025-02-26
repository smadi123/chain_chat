from langchain_ollama.chat_models import ChatOllama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

# Initialize a single LLM - deepseek tends to have better multilingual capabilities
llm = ChatOllama(
    model="deepseek-r1:8b", 
    base_url="http://backend:11434",
    streaming=True
)

# Create an all-in-one prompt that handles translation and response
all_in_one_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an advanced AI assistant capable of understanding and responding in both Arabic and English.

Process each user query through these steps:
1. If the input is in Arabic, first understand it as is
2. Generate a thoughtful, informative response
3. If the original query was in Arabic, provide your response in Arabic with appropriate emojis

Important:
- Your response should be direct and concise
- Include emojis in Arabic responses to enhance readability
- Maintain the same RTL formatting as the user's input for Arabic
- Stream your response token by token as you generate it"""
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

# Create the chain
chain = all_in_one_prompt | llm

# Set up history
history = StreamlitChatMessageHistory()

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
        full_response = ""
        
        # Stream the response directly to the user
        for chunk in chain.stream({
            "input": question,
            "chat_history": history.messages
        }):
            if hasattr(chunk, 'content'):
                content_chunk = chunk.content
                full_response += content_chunk
                response_container.markdown(
                    f'<div style="text-align: right; direction: rtl;">{full_response}</div>',
                    unsafe_allow_html=True
                )
        
        # Add assistant response to history
        history.add_ai_message(full_response)