from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import Runnable
import os
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),
    temperature=0.0,
    top_k=1,
    top_p=1.0,
    do_sample=False,
    repetition_penalty=1.0,
)

model=ChatHuggingFace(llm=llm)

# Define output parser to return JSON
parser = JsonOutputParser()

# LangChain Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", 
     """You are an intelligent assistant that expands the query only when its vague insurance or policy-related queries 
     and provides a reasoning path for document retrieval and decision making.
     
     Given a natural language query, do two things:
     1. Expand the query with relevant structure by extracting these information if available (age, location, treatment, policy duration).
     2. Generate a chain of reasoning steps that explain how a decision could be made using policy clauses.
     
     Return JSON with two fields: 
     - "expanded_query" (string)
     - "thought_steps" (string with bullet points or step-by-step reasoning)"""),
     
    ("human", "{query}")
])

# Chain: Prompt → LLM → JSON Parser
query_expansion_chain: Runnable = prompt | model | parser


def expand_query_and_thought(query: str) -> dict:
    """
    Run the LangChain query expansion and CoT generation.
    :param query: user input string
    :return: dict with expanded_query and thought_steps
    """
    try:
        return query_expansion_chain.invoke({"query": query})
    except Exception as e:
        return {
            "expanded_query": query,
            "thought_steps": "Chain-of-thought could not be generated."
        }
