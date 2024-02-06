import time
start_time = time.time()



import os

os.environ["OPENAI_API_KEY"] = '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'
os.environ["HUGGINGFACEHUB_API_TOKEN"]='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'

import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_community.document_loaders import ArxivLoader


from langchain_community.llms import Ollama


#check is mine works: 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
#Pawel: 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

# os.environ["HUGGINGFACEHUB_API_TOKEN"] = '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'


from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


llama = "llama2"
openAi = False


docs = ArxivLoader(query="2202.10488", load_max_docs=2).load() #TODO load more 
#2202.10488
#2012.06566
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20) #TODO: play around with those parameters, perhaps larger is 
splits = text_splitter.split_documents(docs)


if (openAi):
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings()) # make sure that's allowed
else:
    oembed = OllamaEmbeddings(base_url="http://localhost:11434", model=llama)
    vectorstore = Chroma.from_documents(documents=splits, embedding=oembed)#embeddings)#OllamaEmbeddings



#this works more or less but I don't know if that's correct::

#this is more natural, but doesn't work so far:
# embeddings = OllamaEmbeddings()


print(len(splits))

n = 10

splits = splits







# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")


llm = Ollama(model=llama, temperature=0.1)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.output_parsers import ResponseSchema, StructuredOutputParser


response_schemas = [
    ResponseSchema(name="question", description="generated question"),
    ResponseSchema(
        name="correct answer",
        description="correct answer to generated question",
    ),
    ResponseSchema(
        name="incorrect answer 1",
        description="incorrect answer to generated question",
    ),
    ResponseSchema(
        name="incorrect answer 2",
        description="Second incorrect answer to generated question",
    ),
]

#quiz with closed questions, one for every chapter of the paper in context.

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

template = """ You are an academic teacher preparing graduation test based on the paper.
Use the following pieces of context to generate one question.
Highlight which of the answers is correct.
The quiz should be difficult, concerning details of the paper.



{context}

Question: {question}

Quiz:"""
custom_rag_prompt = PromptTemplate.from_template(template, partial_variables={"format_instructions": format_instructions},)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser()
)


# rag_chain.invoke("What are twin stau interactions in direct detection experiments?")

print("--- %s seconds ---" % (time.time() - start_time))

for chunk in rag_chain.stream("Prepare a quiz with question on twin stau"):
    print(chunk, end="", flush=True)