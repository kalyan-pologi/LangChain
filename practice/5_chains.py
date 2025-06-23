# --------------------- Imports ---------------------
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

# --------------------- Load .env ---------------------
load_dotenv()

# --------------------- Models ---------------------
openai_model = ChatOpenAI()  # Default GPT-3.5
anthropic_model = ChatAnthropic(model_name='claude-3-7-sonnet-20250219')

# --------------------- Output Parsers ---------------------
str_parser = StrOutputParser()

# Pydantic model for structured output parsing
class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description='Classified sentiment of the feedback.')

structured_parser = PydanticOutputParser(pydantic_object=Feedback)

# --------------------- Feedback Classifier + Branch Response ---------------------

feedback_prompt = PromptTemplate(
    template=(
        "Classify the sentiment of the following feedback text into positive or negative:\n"
        "{feedback}\n{format_instruction}"
    ),
    input_variables=["feedback"],
    partial_variables={"format_instruction": structured_parser.get_format_instructions()}
)

# Prompts for response generation
positive_response_prompt = PromptTemplate(
    template="Write an appropriate response to this positive feedback:\n{feedback}",
    input_variables=["feedback"]
)

negative_response_prompt = PromptTemplate(
    template="Write an appropriate response to this negative feedback:\n{feedback}",
    input_variables=["feedback"]
)

# Chain for classification
classifier_chain = feedback_prompt | openai_model | structured_parser

# Conditional branching based on sentiment
feedback_response_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', positive_response_prompt | openai_model | str_parser),
    (lambda x: x.sentiment == 'negative', negative_response_prompt | openai_model | str_parser),
    RunnableLambda(lambda x: "Could not classify sentiment.")
)

# Final composed chain
feedback_chain = classifier_chain | feedback_response_chain

# Run sentiment logic
print("\n🎯 Sentiment Analysis + Response Generation:")
print(feedback_chain.invoke({"feedback": "This is a beautiful phone"}))
feedback_chain.get_graph().print_ascii()

# --------------------- Parallel Chain: Notes + Quiz + Merge ---------------------

# Prompts for generating notes and quiz
notes_prompt = PromptTemplate(
    template="Generate short and simple notes from the following text:\n{text}",
    input_variables=["text"]
)

quiz_prompt = PromptTemplate(
    template="Generate 5 short question answers from the following text:\n{text}",
    input_variables=["text"]
)

merge_prompt = PromptTemplate(
    template="Merge the provided notes and quiz into a single document.\nNotes:\n{notes}\n\nQuiz:\n{quiz}",
    input_variables=["notes", "quiz"]
)

# Run notes with OpenAI, quiz with Anthropic
parallel_chain = RunnableParallel({
    "notes": notes_prompt | openai_model | str_parser,
    "quiz": quiz_prompt | anthropic_model | str_parser
})

# Merge chain
merge_chain = merge_prompt | openai_model | str_parser

# Combine everything
svm_text = """
Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression, and outlier detection.

Advantages:
- Effective in high-dimensional spaces.
- Still effective when the number of dimensions is greater than the number of samples.
- Uses support vectors, so it's memory efficient.
- Supports custom and common kernels.

Disadvantages:
- Overfitting possible in high feature-to-sample ratios.
- No direct probability output (requires cross-validation).
"""

combined_chain = parallel_chain | merge_chain

print("\n📚 Parallel Notes + Quiz + Merge:")
print(combined_chain.invoke({"text": svm_text}))
combined_chain.get_graph().print_ascii()

# --------------------- Simple Topic-Based Report + Summary Chain ---------------------

topic_prompt = PromptTemplate(
    template="Generate a detailed report on {topic}",
    input_variables=["topic"]
)

summary_prompt = PromptTemplate(
    template="Generate a 5-pointer summary from the following text:\n{text}",
    input_variables=["text"]
)

report_chain = topic_prompt | openai_model | str_parser | summary_prompt | openai_model | str_parser

print("\n📊 Topic Report and Summary:")
print(report_chain.invoke({"topic": "Unemployment in India"}))
report_chain.get_graph().print_ascii()

# --------------------- Simple Fact Generator ---------------------

fact_prompt = PromptTemplate(
    template="Generate 5 interesting facts about {topic}",
    input_variables=["topic"]
)

facts_chain = fact_prompt | openai_model | str_parser

print("\n🌟 Fun Fact Generator:")
print(facts_chain.invoke({"topic": "cricket"}))
facts_chain.get_graph().print_ascii()
