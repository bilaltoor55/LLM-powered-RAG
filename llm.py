from operator import itemgetter

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import get_buffer_string
from langchain_core.prompts import format_document
from langchain.prompts.prompt import PromptTemplate
import re

# Define prompts for condensing questions and generating answers
condense_question = """Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question.

Chat History:
{chat_history}

Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_question)

answer = """
### Instruction:
You're a helpful research assistant who answers questions based on provided research in a clear and easy-to-understand way.
If there is no relevant research or the research is irrelevant, reply that you can't answer.
Please reply with just the detailed answer and your sources. If you're unable to answer the question, do not list sources.

## Research:
{context}

## Question:
{question}
"""
ANSWER_PROMPT = ChatPromptTemplate.from_template(answer)

# Default document formatting template
DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(
    template="Source Document: {source}, Page {page}:\n{page_content}"
)


def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    """
    Combines retrieved documents into a formatted string.

    Args:
        docs (list): List of documents.
        document_prompt: Prompt template for document formatting.
        document_separator (str): Separator between documents.

    Returns:
        str: Combined and formatted document string.
    """
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


# Memory setup
memory = ConversationBufferMemory(
    return_messages=True, output_key="answer", input_key="question"
)


def clean_response_chunk(chunk):
    """
    Cleans response chunks by removing unwanted spaces and newlines.

    Args:
        chunk (str): Raw response chunk.

    Returns:
        str: Cleaned response chunk.
    """
    return re.sub(r'\s+', ' ', chunk.strip())


def getStreamingChain(question: str, memory, llm, db):
    """
    Generates a streaming chain for conversational retrieval.

    Args:
        question (str): The user query.
        memory (ConversationBufferMemory): Memory object for context.
        llm (object): Language model instance.
        db (object): Vector database for retrieving documents.

    Returns:
        generator: A streaming generator yielding response chunks.
    """
    if db is None:
        raise ValueError("Database is not initialized. Please index documents first.")

    retriever = db.as_retriever(search_kwargs={"k": 10})

    # Set up memory and standalone question generation
    loaded_memory = RunnablePassthrough.assign(
        chat_history=RunnableLambda(
            lambda x: "\n".join(
                [f"{item['role']}: {item['content']}" for item in x["memory"]]
            )
        )
    )

    standalone_question = {
        "standalone_question": {
            "question": lambda x: x["question"],
            "chat_history": lambda x: x["chat_history"],
        }
        | CONDENSE_QUESTION_PROMPT
        | llm
    }

    retrieved_documents = {
        "docs": itemgetter("standalone_question") | retriever,
        "question": lambda x: x["standalone_question"],
    }

    final_inputs = {
        "context": lambda x: _combine_documents(x["docs"]),
        "question": itemgetter("question"),
    }

    answer = final_inputs | ANSWER_PROMPT | llm

    final_chain = loaded_memory | standalone_question | retrieved_documents | answer

    # Stream the cleaned response
    for chunk in final_chain.stream({"question": question, "memory": memory}):
        yield clean_response_chunk(chunk)


def getChatChain(llm, db):
    """
    Returns a callable chat chain for interactive Q&A.

    Args:
        llm (object): Language model instance.
        db (object): Vector database for document retrieval.

    Returns:
        callable: A chat function for question-answering with context.
    """
    if db is None:
        raise ValueError("Database is not initialized. Please index documents first.")

    retriever = db.as_retriever(search_kwargs={"k": 10})

    loaded_memory = RunnablePassthrough.assign(
        chat_history=RunnableLambda(memory.load_memory_variables)
        | itemgetter("history"),
    )

    standalone_question = {
        "standalone_question": {
            "question": lambda x: x["question"],
            "chat_history": lambda x: get_buffer_string(x["chat_history"]),
        }
        | CONDENSE_QUESTION_PROMPT
        | llm
    }

    retrieved_documents = {
        "docs": itemgetter("standalone_question") | retriever,
        "question": lambda x: x["standalone_question"],
    }

    final_inputs = {
        "context": lambda x: _combine_documents(x["docs"]),
        "question": itemgetter("question"),
    }

    answer = {
        "answer": final_inputs
        | ANSWER_PROMPT
        | llm.with_config(callbacks=[StreamingStdOutCallbackHandler()]),
        "docs": itemgetter("docs"),
    }

    final_chain = loaded_memory | standalone_question | retrieved_documents | answer

    def chat(question: str):
        """
        Processes a question, retrieves relevant context, and returns the answer.

        Args:
            question (str): User query.

        Returns:
            dict: Contains the final answer and retrieved documents.
        """
        inputs = {"question": question}
        result = final_chain.invoke(inputs)
        memory.save_context(inputs, {"answer": result["answer"]})
        return result

    return chat
