from dotenv import load_dotenv
import os
from tavily import TavilyClient

load_dotenv()

API_KEY = os.getenv("TAVILLY_API_KEY")

CLIENT = TavilyClient(api_key=API_KEY)

def web_search(query: str, num_results: int = 5):
    results = CLIENT.search(query=query)
    return results['results'][:num_results]

def in_detail_search(url: str):
    detailed_info = CLIENT.extract(urls=[url])
    return detailed_info['results'][0]['raw_content']


