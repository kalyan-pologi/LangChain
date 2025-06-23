# -------------------- Imports --------------------
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import (
    RunnableSequence,
    RunnableParallel,
    RunnablePassthrough,
    RunnableBranch,
    RunnableLambda
)

load_dotenv()
model = ChatOpenAI()
parser = StrOutputParser()

# -------------------- 1. Conditional Summary Based on Length --------------------
print("\n🧾 Conditional Summary Example")

prompt_report = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)

prompt_summary = PromptTemplate(
    template="Summarize the following text \n {text}",
    input_variables=["text"]
)

report_chain = prompt_report | model | parser

summary_branch = RunnableBranch(
    (lambda x: len(x.split()) > 300, prompt_summary | model | parser),
    RunnablePassthrough()
)

conditional_chain = RunnableSequence(report_chain, summary_branch)
print(conditional_chain.invoke({'topic': 'Russia vs Ukraine'}))

# -------------------- 2. Joke with Word Count (Parallel Output) --------------------
print("\n😂 Joke + Word Count Example")

prompt_joke = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=["topic"]
)

def word_count(text):
    return len(text.split())

joke_chain = RunnableSequence(prompt_joke, model, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_count': RunnableLambda(word_count)
})

combined_chain = RunnableSequence(joke_chain, parallel_chain)

joke_result = combined_chain.invoke({'topic': 'AI'})
print(f"{joke_result['joke']} \nWord Count: {joke_result['word_count']}")

# -------------------- 3. Multi-Platform Content (Tweet & LinkedIn) --------------------
print("\n📱 Tweet + LinkedIn Post Example")

tweet_prompt = PromptTemplate(
    template="Generate a tweet about {topic}",
    input_variables=["topic"]
)

linkedin_prompt = PromptTemplate(
    template="Generate a Linkedin post about {topic}",
    input_variables=["topic"]
)

content_parallel = RunnableParallel({
    'tweet': RunnableSequence(tweet_prompt, model, parser),
    'linkedin': RunnableSequence(linkedin_prompt, model, parser)
})

social_result = content_parallel.invoke({'topic': 'AI'})
print("Tweet:", social_result['tweet'])
print("LinkedIn:", social_result['linkedin'])

# -------------------- 4. Joke + Explanation (Parallel Insight) --------------------
print("\n🧠 Joke + Explanation (Parallel)")

explanation_prompt = PromptTemplate(
    template="Explain the following joke - {text}",
    input_variables=["text"]
)

joke_chain = RunnableSequence(prompt_joke, model, parser)

joke_parallel = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation': RunnableSequence(explanation_prompt, model, parser)
})

full_joke_explain_chain = RunnableSequence(joke_chain, joke_parallel)

print(full_joke_explain_chain.invoke({'topic': 'cricket'}))

# -------------------- 5. Joke → Explain (Sequential) --------------------
print("\n🔁 Joke → Explanation (Sequential Chain)")

joke_explanation_chain = RunnableSequence(
    prompt_joke, model, parser,
    explanation_prompt, model, parser
)

print(joke_explanation_chain.invoke({'topic': 'AI'}))
