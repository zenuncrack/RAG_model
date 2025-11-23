import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Sequence
from operator import add as add_messages

from langgraph.graph import StateGraph, END
from langchain_core.messages import (BaseMessage, SystemMessage, HumanMessage, ToolMessage)

from langchain_groq import ChatGroq
from langchain_core.tools import tool

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults







def build_rag_agent_from_file(file_path):
    llm_base = ChatGroq(model="llama-3.3-70b-versatile", temperature=1)
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)

    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(pages)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = Chroma.from_documents(docs, embeddings)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

  
    tavily = TavilySearchResults(k=3)

    @tool
    def web_search_tool(query: str) -> str:
        """Searches the web for fresh information using Tavily."""
        try:
           results = tavily.invoke({"query": query})
           return str(results)
        except Exception as e:
           return f"Web search failed: {e}"

    @tool
    def retriever_tool(query: str) -> str:
        """Retrieve answers from the uploaded file."""
        docs = retriever.invoke(query)

        if not docs:
            return "No relevant information found in the uploaded document."

        chunks = []
        for i, d in enumerate(docs):
            chunks.append(f"[Chunk {i+1}]\n{d.page_content}")

        return "\n\n".join(chunks)

    tools = [retriever_tool, web_search_tool]


    llm = llm_base.bind_tools(tools)

    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]

    def should_continue(state: AgentState):
        result = state['messages'][-1]
        return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0

    system_prompt = """
    You are an intelligent RAG assistant.
    Use retriever_tool to search the uploaded document.
    Use web_search_tool to search the web when needed.
    Cite chunks when answering.
    """

    tools_dict = {t.name: t for t in tools}

    def call_llm(state: AgentState) -> AgentState:
        messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
        message = llm.invoke(messages)
        return {"messages": [message]}

    def take_action(state: AgentState) -> AgentState:
        tool_calls = state["messages"][-1].tool_calls
        results = []

        for t in tool_calls:
            tool_name = t["name"]
            query = t["args"].get("query", "")

            if tool_name not in tools_dict:
                result = f"Tool {tool_name} not found."
            else:
                result = tools_dict[t['name']].invoke(t['args'].get('query', ''))


            results.append(
                ToolMessage(
                    tool_call_id=t["id"],
                    name=tool_name,
                    content=str(result)
                )
            )

        return {"messages": results}

    graph = StateGraph(AgentState)
    graph.add_node("llm", call_llm)
    graph.add_node("retriever", take_action)

    graph.add_conditional_edges(
        "llm",
        should_continue,
        {True: "retriever", False: END}
    )

    graph.add_edge("retriever", "llm")
    graph.set_entry_point("llm")

    return graph.compile()
