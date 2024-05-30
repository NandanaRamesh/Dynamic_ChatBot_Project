import requests
from bs4 import BeautifulSoup
from transformers import pipeline
from urllib.parse import urljoin
import time


#The main function which is used to scrape website content from the website
def scrape_website(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        main_content = []
        for tag in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']):
            main_content.append(tag.get_text(separator=' ', strip=True))

        return ' '.join(main_content)
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return None


# This one is used to find all links on the page and scrape them, and I can keep track of the visited ones
def scrape_multiple_pages(start_url, max_pages=1):
    visited = set()
    to_visit = [start_url]
    content = []

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue

        page_content = scrape_website(url)
        if page_content:
            content.append(page_content)

        visited.add(url)

        # For finding the other linked pages in the website
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            for a_tag in soup.find_all('a', href=True):
                link = urljoin(url, a_tag['href'])
                if link not in visited and link.startswith(start_url):
                    to_visit.append(link)
        except requests.RequestException as e:
            print(f"Failed to retrieve links from {url}: {e}")

        # To avoid overwhelming the server for good practice
        time.sleep(1)

    return ' '.join(content)


qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")


# Answering questions from the website content
def answer_question_from_website(content, question):
    if content:
        try:
            result = qa_pipeline({
                'context': content,
                'question': question
            })
            return result['answer']
        except Exception as e:
            return f"An error occurred during the QA process....{e}"
    else:
        return "Sorry, I couldn't retrieve the information from the website."


if __name__ == "__main__":
    start_url = 'https://en.wikipedia.org/wiki/NCT_(group)'
    content = scrape_multiple_pages(start_url)

    if not content:
        print("Failed to retrieve content from the website.")
    else:
        while True:
            user_question = input("Hi! How may I help you? (Type 'Bye' to exit) ")
            if user_question.lower() == "bye":
                print("Goodbye!")
                break
            answer = answer_question_from_website(content, user_question)
            print(answer)
