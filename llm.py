from operator import itemgetter

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import get_buffer_string
from langchain_core.prompts import format_document
from langchain.prompts.prompt import PromptTemplate


condense_question = """Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question.

Chat History:
{chat_history}

Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_question)

answer = """
### Instruction:
You're a helpful research assistant, who answers questions based on provided research in a clear way and easy-to-understand way.
If there is no research, or the research is irrelevant to answering the question, simply reply that you can't answer.
Please reply with just the detailed answer and your sources. If you're unable to answer the question, do not list sources

## Research:
{context}

## Question:
{question}
"""
ANSWER_PROMPT = ChatPromptTemplate.from_template(answer)

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(
    template="Source Document: {source}, Page {page}:\n{page_content}"
)


def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


memory = ConversationBufferMemory(
    return_messages=True, output_key="answer", input_key="question"
)


def getStreamingChain(question: str, memory, llm, db):
    """
    Generates a streaming chain for conversational retrieval.

    Args:
        question (str): The user query.
        memory (ConversationBufferMemory): The memory object to maintain context.
        llm (object): The language model to use for response generation.
        db (object): The vector database for retrieving documents.

    Returns:
        generator: A streaming generator yielding response chunks.
    """
    # Check if the database is initialized
    if db is None:
        raise ValueError("Database is not initialized. Please index documents first.")

    # Retrieve relevant documents
    retriever = db.as_retriever(search_kwargs={"k": 10})

    # Set up memory and standalone question generation
    loaded_memory = RunnablePassthrough.assign(
        chat_history=RunnableLambda(
            lambda x: "\n".join(
                [f"{item['role']}: {item['content']}" for item in x["memory"]]
            )
        ),
    )

    standalone_question = {
        "standalone_question": {
            "question": lambda x: x["question"],
            "chat_history": lambda x: x["chat_history"],
        }
        | CONDENSE_QUESTION_PROMPT
        | llm
    }

    # Retrieve documents and prepare inputs
    retrieved_documents = {
        "docs": itemgetter("standalone_question") | retriever,
        "question": lambda x: x["standalone_question"],
    }

    final_inputs = {
        "context": lambda x: _combine_documents(x["docs"]),
        "question": itemgetter("question"),
    }

    # Generate answers using the prompt and LLM
    answer = final_inputs | ANSWER_PROMPT | llm

    # Assemble the final chain
    final_chain = loaded_memory | standalone_question | retrieved_documents | answer

    # Return the streamed response
    return final_chain.stream({"question": question, "memory": memory})


def getChatChain(llm, db):
    """
    Returns a callable chat chain for interactive question-answering.

    Args:
        llm (object): The language model instance.
        db (object): The vector database for retrieving documents.

    Returns:
        callable: A chat function that takes a question as input and maintains context.
    """
    # Check if the database is initialized
    if db is None:
        raise ValueError("Database is not initialized. Please index documents first.")

    retriever = db.as_retriever(search_kwargs={"k": 10})

    # Set up memory and standalone question generation
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

    # Retrieve documents and prepare inputs
    retrieved_documents = {
        "docs": itemgetter("standalone_question") | retriever,
        "question": lambda x: x["standalone_question"],
    }

    # Generate answers using the prompt and LLM
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

    # Assemble the final chain
    final_chain = loaded_memory | standalone_question | retrieved_documents | answer

    def chat(question: str):
        """
        Processes a question, retrieves relevant context, and returns the answer.

        Args:
            question (str): The user query.

        Returns:
            None
        """
        inputs = {"question": question}
        result = final_chain.invoke(inputs)
        memory.save_context(inputs, {"answer": result["answer"]})

    return chat
