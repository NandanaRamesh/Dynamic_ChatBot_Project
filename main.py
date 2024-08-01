import requests
from bs4 import BeautifulSoup
from transformers import pipeline

# Improved web scraping with prioritized section extraction
def scrape_website(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Prioritize relevant sections
        main_content = []
        for tag in soup.find_all(['h1', 'h2', 'h3', 'p']):
            main_content.append(tag.get_text(separator=' ', strip=True))

        return ' '.join(main_content)
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return None

# Only scrape the provided URL, without following links
def scrape_single_page(start_url):
    page_content = scrape_website(start_url)
    return page_content if page_content else ""

# Use a more advanced model for better accuracy
qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

def answer_question_from_website(content, question):
    if content:
        try:
            # Chunk content if too large
            max_chunk_size = 512  # BERT-like models usually have a max token size of 512
            chunks = [content[i:i + max_chunk_size] for i in range(0, len(content), max_chunk_size)]
            answers = []

            for chunk in chunks:
                result = qa_pipeline({
                    'context': chunk,
                    'question': question
                })
                answers.append((result['answer'], result['score']))

            # Return the answer with the highest score
            best_answer = max(answers, key=lambda x: x[1])[0]
            return best_answer
        except Exception as e:
            return f"An error occurred during the QA process: {e}"
    else:
        return "Sorry, I couldn't retrieve the information from the website."

if __name__ == "__main__":
    start_url = 'https://en.wikipedia.org/wiki/Data_science'
    content = scrape_single_page(start_url)

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
