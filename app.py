import os
import sys
import logging
import operator
from typing import TypedDict, Annotated, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from dotenv import load_dotenv

import uvicorn

from langchain_mongodb import MongoDBChatMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("HSE-API-ASYNC")

app = FastAPI(title="HSE Agent Async API")
INDEX_NAME = "hse-sop-ultimate-index"

if not os.getenv("OPENAI_API_KEY") or not os.getenv("PINECONE_API_KEY") or not os.getenv("MONGO_URI"):
    logger.critical("❌ FATAL: Missing API Keys or MONGO_URI.")
    sys.exit(1)

class ChatRequest(BaseModel):
    message: str
    session_id: str

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

def initialize_graph():
    try: 
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vector_store = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
        llm = ChatOpenAI(model="gpt-4o", temperature=0)

        @tool(response_format="content_and_artifact")
        def retrieve_sop_info(query: str):
            """Retrieve HSE SOP documents for the given query."""
            try:
                docs = vector_store.similarity_search(query, k=20)
                serialized = "\n\n".join(
                    (f"Source: {doc.metadata.get('source', 'Unknown')} (Page {doc.metadata.get('page', '?')})\nContent: {doc.page_content}")
                    for doc in docs
                )
                return serialized, docs
            except Exception as e:
                logger.error(f"❌ Tool Error: {e}")
                return "Error retrieving documents.", []

        tools = [retrieve_sop_info]
        llm_with_tools = llm.bind_tools(tools)

        def agent_node(state):
            return {"messages": [llm_with_tools.invoke(state["messages"])]}

        tool_node = ToolNode(tools)

        def should_continue(state):
            last_message = state["messages"][-1]
            if last_message.tool_calls:
                return "tools"
            return END

        workflow = StateGraph(AgentState)
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", tool_node)
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges("agent", should_continue)
        workflow.add_edge("tools", "agent")

        return workflow.compile()
    except Exception as e:
        logger.critical(f"❌ Failed to initialize Agent Graph: {e}")
        sys.exit(1)

agent_app = initialize_graph()

SYSTEM_PROMPT = SystemMessage(content=(
    "You are a specialized HSE (Health, Safety, Environment) Guardian. "
    "Use 'retrieve_sop_info' for factual questions. Do not hallucinate."
))

def get_history_sync(session_id: str):
    return MongoDBChatMessageHistory(
        session_id=session_id,
        connection_string=os.getenv("MONGO_URI"),
        database_name="hse_agent_db",
        collection_name="chat_history"
    )

@app.get('/health')
async def health_check():
    return {"status": "healthy"}

@app.post('/chat')
async def chat_endpoint(request: ChatRequest):
    try:
        history = await run_in_threadpool(get_history_sync, request.session_id)

        messages = [SYSTEM_PROMPT] + history.messages + [HumanMessage(content=request.message)]

        response = await agent_app.ainvoke({"messages": messages})
        final_content = response["messages"][-1].content

        def save_interaction():
            history.add_user_message(request.message)
            history.add_ai_message(final_content)
        
        await run_in_threadpool(save_interaction)

        return {
            "response": final_content,
            "session_id": request.session_id
        }

    except Exception as e:
        logger.error(f"❌ Error processing chat: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == '__main__':
    print("🚀 Starting Async Server...")
    uvicorn.run(app, host='0.0.0.0', port=5000)