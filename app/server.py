import os
from typing import Dict, Any

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores.supabase import SupabaseVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, ConfigurableField
from supabase.client import create_client
from dotenv import dotenv_values

# Add environment variables
env_vars = dotenv_values(".env")
os.environ["OPENAI_API_KEY"] = env_vars['OPENAI_API_KEY']
os.environ["LANGCHAIN_API_KEY"] = env_vars['LANGCHAIN_API_KEY']
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Connect to Supabase
supabase_url = env_vars['SUPABASE_URL']
supabase_key = env_vars['SUPABASE_KEY']
supabase = create_client(supabase_url, supabase_key)

# Use OpenAI embeddings and define vectorstore
embeddings = OpenAIEmbeddings()

vectorstore = SupabaseVectorStore(
    client=supabase,
    embedding=embeddings,
    table_name="documents",
    query_name="match_documents",
)

# Use vectorstore as a retriever
retriever = vectorstore.as_retriever()

# Add configurability of search kwargs at runtime.
configurable_retriever = retriever.configurable_fields(
    search_kwargs=ConfigurableField(
        id="search_kwargs",
        name="Search Kwargs",
        description="The search kwargs to use",
    )
)

# Utility function to format retieved documents for the prompt.
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Construct prompt and chain
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI()

chain = (
    RunnableParallel({"context": configurable_retriever | format_docs, "question": RunnablePassthrough()})
    | prompt
    | model
    | StrOutputParser()
)

### THE API ###
app = FastAPI()


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


add_routes(app, chain, path="/rag-supabase")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
