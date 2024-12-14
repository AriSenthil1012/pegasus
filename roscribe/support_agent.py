import sys 
import os 

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.memory import ConversationBufferMemory

# Add the parent directory (pegasus) to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from roscribe.tools import get_rag_tools, get_file_tool
from roscribe.prompts import get_support_agent_prompt

print("hello")