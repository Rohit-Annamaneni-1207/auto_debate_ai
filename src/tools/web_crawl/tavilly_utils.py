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
    if detailed_info['results']:
        return detailed_info['results'][0]['raw_content']
    return ""


if __name__ == "__main__":
    # Example usage
    search_results = web_search("Artificial Intelligence", num_results=3)
    for i, result in enumerate(search_results):
        print(f"Result {i+1}:")
        print(f"Title: {result['title']}")
        print(f"URL: {result['url']}")

    if search_results:
        first_url = search_results[1]['url']
        detailed_content = in_detail_search(first_url)
        print(f"Detailed content from {first_url}:\n{detailed_content}")  # Print first 500 characters


