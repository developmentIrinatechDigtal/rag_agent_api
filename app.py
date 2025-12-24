import os
import sys
import logging
import operator
from typing import TypedDict, Annotated, List

from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import openai

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("HSE-API")

app = Flask(__name__)
INDEX_NAME = "hse-sop-ultimate-index"

if not os.getenv("OPENAI_API_KEY") or not os.getenv("PINECONE_API_KEY"):
    logger.critical("❌ FATAL: Missing API Keys in environment variables.")
    sys.exit(1)

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

def initialize_graph():
    try: 
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vector_store = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
        
        llm = ChatOpenAI(model="gpt-4o", temperature=0)

        @tool(response_format="content_and_artifact")
        def retrieve_sop_info(query: str):
            """Retrieves HSE SOP info. Do NOT answer from memory."""
            logger.info(f"🔎 Tool Query: {query}")
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

        logger.info("✅ Graph initialized successfully.")
        return workflow.compile()

    except Exception as e:
        logger.critical(f"❌ Failed to initialize Agent Graph: {e}")
        sys.exit(1)

agent_app = initialize_graph()

SYSTEM_PROMPT = SystemMessage(content=(
    "You are a specialized HSE (Health, Safety, Environment) Guardian. "
    "Use 'retrieve_sop_info' for factual questions. Do not hallucinate."
))

@app.errorhandler(400)
def bad_request(e):
    return jsonify({"error": "Bad Request", "message": str(e)}), 400

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not Found", "message": "Endpoint does not exist."}), 404

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"🔥 500 Error: {e}")
    return jsonify({"error": "Internal Server Error", "message": "An unexpected error occurred."}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400
    
    data = request.get_json()
    user_message = data.get('message')
    history_data = data.get('history', [])

    if not user_message:
        return jsonify({"error": "Missing 'message' field"}), 400

    try:
        messages = [SYSTEM_PROMPT]
        for msg in history_data:
            role = msg.get('role')
            content = msg.get('content')
            if role == 'user' and content:
                messages.append(HumanMessage(content=content))
            elif role == 'assistant' and content:
                messages.append(AIMessage(content=content))
        
        messages.append(HumanMessage(content=user_message))

    except Exception as e:
        logger.error(f"Error parsing history: {e}")
        return jsonify({"error": "Invalid history format"}), 400

    try:
        response = agent_app.invoke({"messages": messages})
        final_content = response["messages"][-1].content
        return jsonify({"response": final_content})

    except openai.RateLimitError:
        logger.error("❌ OpenAI Rate Limit Exceeded")
        return jsonify({"error": "Service overloaded. Please try again later."}), 503

    except openai.AuthenticationError:
        logger.critical("❌ OpenAI Authentication Failed")
        return jsonify({"error": "Internal authentication error."}), 500

    except Exception as e:
        logger.error(f"❌ Runtime Error in Graph: {e}")
        return jsonify({"error": "An error occurred while processing the request."}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)