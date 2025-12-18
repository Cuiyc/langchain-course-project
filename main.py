from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
#from langchain_openai import ChatOpenAI


from langchain_google_genai import ChatGoogleGenerativeAI

from tavily import TavilyClient

tavily = TavilyClient()


@tool
def search(query: str ) -> str:
    """
    Make internet search for the query
    
    :param query: anything to search from internet
    :type query: str
    :return: query result
    :rtype: str
    """
    print(f"Searching for {query}")
    #return "Tokyo weather is sunny"
    return tavily.search(query = query)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=1.0,  # Gemini 3.0+ defaults to 1.0
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)
tools = [search]
agent = create_agent(model=llm, tools=tools)


def main():
    print("Hello from langchain-course!")
    result = agent.invoke(input={"messages": [HumanMessage("search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details")]})
    print(result)




if __name__ == "__main__":
    main()
