import requests
from bs4 import BeautifulSoup

# Your Hugging Face API key
API_KEY = "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"
headers = {"Authorization": f"Bearer {API_KEY}"}

def scrape_website(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # This helps to prioritize relevant sections
        main_content = []
        for tag in soup.find_all(['h1', 'h2', 'h3', 'p']):
            main_content.append(tag.get_text(separator=' ', strip=True))

        return ' '.join(main_content)
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return None

# Only scraping a single page of the provided URL, without following links
def scrape_single_page(start_url):
    page_content = scrape_website(start_url)
    return page_content if page_content else ""

def ask_gpt_j(question, context):
    api_url = "https://api-inference.huggingface.co/models/EleutherAI/gpt-j-6B"
    # Combining the context and question into a single input
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    data = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 150, "return_full_text": False}
    }

    response = requests.post(api_url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()[0]['generated_text']
    else:
        raise Exception(f"Request failed with status code {response.status_code}: {response.text}")

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
            try:
                answer = ask_gpt_j(user_question, content)
                print(answer)
            except Exception as e:
                print(f"An error occurred: {e}")
