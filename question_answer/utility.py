import torch
from transformers import AutoTokenizer, AutoModel
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from functools import partial
from decouple import config

persistent_directory = config('PERSISTENT_DIRECTORY')

api_key = config('OPEN_AI_API_KEY')
BASE_TRANSCRIPT_PATH = config('BASE_TRANSCRIPT_PATH')


def valid_integer(value):
    """
    Validates if the provided value is a valid integer.

    Args:
    - `value` (str): The value to be validated.

    Returns:
    - bool: True if the value is a valid integer, False otherwise.
    """
    # valid_integer changes
    if not value.isdigit():
        return False
    return True


# Function to compute MaxSim
def maxsim(query_embedding, document_embedding):
    # Expand dimensions for broadcasting
    # Query: [batch_size, query_length, embedding_size] -> [batch_size, query_length, 1, embedding_size]
    # Document: [batch_size, doc_length, embedding_size] -> [batch_size, 1, doc_length, embedding_size]
    expanded_query = query_embedding.unsqueeze(2)
    expanded_doc = document_embedding.unsqueeze(1)

    # Compute cosine similarity across the embedding dimension
    sim_matrix = torch.nn.functional.cosine_similarity(expanded_query, expanded_doc, dim=-1)

    # Take the maximum similarity for each query token (across all document tokens)
    # sim_matrix shape: [batch_size, query_length, doc_length]
    max_sim_scores, _ = torch.max(sim_matrix, dim=2)

    # Average these maximum scores across all query tokens
    avg_max_sim = torch.mean(max_sim_scores, dim=1)
    return avg_max_sim


def get_top_k_docs(query, class_id):
    top_k = 3
    scores = []

    # Get the stored vector db
    embedding = OpenAIEmbeddings(api_key=api_key)
    vectordb = Chroma(
        persist_directory=persistent_directory, embedding_function=embedding
    )
    # create a retriever from the vector database
    # retriever = vectordb.as_retriever(search_type="mmr",
    #                                   search_kwargs={"k": 6},
    #                                   filter={"source": "transcription.txt"})
    # relevant_docs = retriever.invoke(query)
    relevant_docs = vectordb.max_marginal_relevance_search(query,
                                                           k=3,
                                                           filter={"source": f"{BASE_TRANSCRIPT_PATH}{class_id}_transcript.txt"})
    print("persistent directory path...............")
    print(persistent_directory)
    print("relevant docs are here...............................................")
    print(relevant_docs)

    # Load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained("colbert-ir/colbertv2.0")
    model = AutoModel.from_pretrained("colbert-ir/colbertv2.0")

    # Encode the query
    query_encoding = tokenizer(query, return_tensors='pt')
    query_embedding = model(**query_encoding).last_hidden_state.mean(dim=1)

    # Get score for each document
    for document in relevant_docs:
        # print(document)
        document_encoding = tokenizer(document.page_content, return_tensors='pt', truncation=True, max_length=512)
        document_embedding = model(**document_encoding).last_hidden_state

        # Calculate MaxSim score
        score = maxsim(query_embedding.unsqueeze(0), document_embedding)
        scores.append({
            "score": score.item(),
            "document": document.page_content,
        })

    # Sort the scores by highest to lowest and print
    sorted_data = sorted(scores, key=lambda x: x['score'], reverse=True)[:top_k]
    print(sorted_data)
    return format_docs([data['document'] for data in sorted_data])


def format_docs(docs):
    return "\n\n".join(doc for doc in docs)


def question_answer(class_id, query):

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3, openai_api_key=api_key)

    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    If the context is empty, just say that you don't know, don't try to make up an answer.
    Always say "thanks for asking!" at the end of the answer.

    context: {context}

    Question: {question}

    Helpful Answer:"""
    custom_rag_prompt = PromptTemplate.from_template(template)

    response = (
            {"context": partial(get_top_k_docs, class_id=class_id), "question": RunnablePassthrough()}
            | custom_rag_prompt
            | llm
            | StrOutputParser()
    )

    return response.invoke(query)
