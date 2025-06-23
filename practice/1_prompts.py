# -------------------- Import Necessary Libraries --------------------
import os
import streamlit as st
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.prompts import PromptTemplate, load_prompt
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# -------------------- Load Environment Variables --------------------
load_dotenv()

# Load HuggingFace API token from .env file
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in environment variables.")

# -------------------- Initialize HuggingFace Chat Model --------------------
# We're using TinyLlama here as the backend model for generating text
llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # You can replace with another compatible model
    task="text-generation",
    pipeline_kwargs={
        "max_length": 512,
        "temperature": 0.7
    }
)

chat_model = ChatHuggingFace(llm=llm)

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Research Summary Tool", layout="centered")
st.header('🧠 Research Paper Summary Tool')

# Dropdown for selecting a research paper
paper_input = st.selectbox(
    "📄 Select Research Paper",
    [
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis"
    ]
)

# Dropdown for explanation style
style_input = st.selectbox(
    "🎨 Select Explanation Style",
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]
)

# Dropdown for explanation length
length_input = st.selectbox(
    "📏 Select Explanation Length",
    ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"]
)

# -------------------- Load Prompt Template --------------------
# This prompt guides the model in how to format the response
# Make sure you have a template.json saved earlier
template = load_prompt('template.json')

# -------------------- On Button Click: Generate Summary --------------------
if st.button('📚 Generate Summary'):
    chain = template | chat_model
    result = chain.invoke({
        'paper_input': paper_input,
        'style_input': style_input,
        'length_input': length_input
    })
    st.subheader("📖 Summary Output")
    st.write(result.content)

# -------------------- Bonus: Simple Prompt Examples Below --------------------

# 💡 EXAMPLE 1: OpenAI Model - Fun Poem
if st.button("📝 Generate Poem (OpenAI GPT-4)"):
    openai_model = ChatOpenAI(model="gpt-4", temperature=1.2)
    poem_result = openai_model.invoke("Write a 5 line poem on cricket")
    st.write(poem_result.content)

# 💡 EXAMPLE 2: Multilingual Greeting Using PromptTemplate
if st.button("🌍 Greet in 5 Languages"):
    greet_model = ChatOpenAI()
    greeting_template = PromptTemplate(
        template='Greet this person in 5 languages. The name of the person is {name}',
        input_variables=['name']
    )
    prompt = greeting_template.invoke({'name': 'Nitish'})
    result = greet_model.invoke(prompt)
    st.write(result.content)

# -------------------- Chat History Example --------------------
# Simulating a chat conversation using messages
if st.checkbox("🗨️ Show Chat Example"):
    chat_model_example = ChatOpenAI()
    messages = [
        SystemMessage(content="You are a helpful assistant"),
        HumanMessage(content="Tell me about LangChain")
    ]
    result = chat_model_example.invoke(messages)
    messages.append(AIMessage(content=result.content))
    st.markdown("**Conversation History:**")
    for msg in messages:
        role = "👤" if isinstance(msg, HumanMessage) else "🤖"
        st.write(f"{role} {msg.content}")
