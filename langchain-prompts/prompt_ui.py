from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
import streamlit as st
import os
from langchain_core.prompts import PromptTemplate,load_prompt


# Load environment variables from a .env file
load_dotenv()

# Check if the API token is loaded
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in environment variables.")


# Initialize the OpenAI chat model
# chat_model = ChatOpenAI(model_name="gpt-4", temperature=0.7)
llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={
        "max_length": 512,
        "temperature": 0.7
    }
)
chat_model = ChatHuggingFace(llm=llm)

st.header('Reasearch Tool')

paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

template = load_prompt('template.json')


if st.button('Summarize'):
    chain = template | chat_model
    result = chain.invoke({
        'paper_input':paper_input,
        'style_input':style_input,
        'length_input':length_input
    })
    st.write(result.content)