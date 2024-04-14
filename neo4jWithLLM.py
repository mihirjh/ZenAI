import os

import os
import httpx
from langchain_openai import AzureChatOpenAI, ChatOpenAI,AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage

from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Tuple, List, Optional
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_community.graphs import Neo4jGraph
from langchain_community.document_loaders import WikipediaLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from neo4j import GraphDatabase
from yfiles_jupyter_graphs import GraphWidget
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_core.runnables import ConfigurableField, RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_core.documents import Document


def get_pdf_text():
    with open(u"C:\\Users\\mihirjh\\OneDrive - AMDOCS\\jsonDiff\\billingCare.txt", 'r') as file:
        text = file.read()
        return text

load_dotenv("mihirjh.env")

url = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
ollama_base_url = os.getenv("OLLAMA_BASE_URL")
embedding_model_name = os.getenv("EMBEDDING_MODEL")
# Remapping for Langchain Neo4j integration
os.environ["NEO4J_URL"] = url

azure_configs = {     
    "azure_endpoint" : "https://dev-mgmt-infra.mihir.azr/v1/hackathon/regions/canadaeast/",
    "openai_api_version" :"2023-05-15",
    "azure_deployment" : "gpt-35",
    "openai_api_key" :"5d57e861530c4f30b60ffggfae432f52", #Sample not real
    "openai_api_type" :"azure",
    "temperature" : "0",
    "http_client": httpx.Client(verify=False) #SSL Disabled
}
llm = AzureChatOpenAI(**azure_configs)
llm_transformer = LLMGraphTransformer(llm=llm)
#result = llm([HumanMessage(content='Tell me about pluto')])
#print(result)

raw_documents = get_pdf_text()
raw_documents = [Document(page_content=get_pdf_text())]
text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
documents = text_splitter.split_documents(raw_documents)

#print(raw_documents[:10])

graph_documents = llm_transformer.convert_to_graph_documents(documents)
print(f"Nodes:{graph_documents[0].nodes}")
print(f"Relationships:{graph_documents[0].relationships}")

# if Neo4j is local, you can go to http://localhost:7474/ to browse the database
neo4j_graph = Neo4jGraph(url=url, username=username, password=password)
#neo4j_graph.add_graph_documents(graph_documents, baseEntityLabel=True,include_source=True)


neo4j_graph.refresh_schema()
print(neo4j_graph.schema)

from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate



chain = GraphCypherQAChain.from_llm(
    llm, graph=neo4j_graph, verbose=True, validate_cypher=True
)

chain.invoke("How many Bussiness component?")


from langchain_community.embeddings import OllamaEmbeddings




vector_index = Neo4jVector.from_existing_graph(
    OllamaEmbeddings(base_url=ollama_base_url, model="llama2"),
    search_type="hybrid",
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding"
)
question = {"question": "Who is architect of Billing Care Highlihts?"}
#vector_index.similarity_search(question)


# Retriever
neo4j_graph.query(
    "CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")

# Extract entities from text
class Entities(BaseModel):
    """Identifying information about entities."""

    names: List[str] = Field(
        ...,
        description="All the person, organization, or business entities that "
        "appear in the text",
    )

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are extracting organization and person entities from the text.",
        ),
        (
            "human",
            "Use the given format to extract information from the following "
            "input: {question}",
        ),
    ]
)

entity_chain = prompt | llm.with_structured_output(Entities)

#print(entity_chain.invoke({"question": "What is Billing Care application?"}).names)

def generate_full_text_query(input: str) -> str:
    """
    Generate a full-text search query for a given input string.

    This function constructs a query string suitable for a full-text search.
    It processes the input string by splitting it into words and appending a
    similarity threshold (~2 changed characters) to each word, then combines
    them using the AND operator. Useful for mapping entities from user questions
    to database values, and allows for some misspelings.
    """
    full_text_query = ""
    words = [el for el in remove_lucene_chars(input).split() if el]
    for word in words[:-1]:
        full_text_query += f" {word}~2 AND"
    full_text_query += f" {words[-1]}~2"
    print("QUERY: "+ full_text_query)
    return full_text_query.strip()

# Fulltext index query
def structured_retriever(question: str) -> str:
    """
    Collects the neighborhood of entities mentioned
    in the question
    """
    result = ""
    entities = entity_chain.invoke({"question": question})
    for entity in entities.names:
        print("Entitiyyyyy: " + entity)
        response = neo4j_graph.query(
            """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
            YIELD node,score
            CALL {
              MATCH (node)-[r:!MENTIONS]->(neighbor)
              RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
              UNION
              MATCH (node)<-[r:!MENTIONS]-(neighbor)
              RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
            }
            RETURN output LIMIT 50
            """,
            {"query": generate_full_text_query(entity)},
        )
        result += "\n".join([el['output'] for el in response])
    return result

#print(structured_retriever({"question": "What is Billing Care application?"}))


