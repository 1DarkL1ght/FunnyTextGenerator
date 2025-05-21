import re
import json
import os
import requests
import csv
from bs4 import BeautifulSoup



from dotenv import load_dotenv

def get_vk_posts(domain: str, count: int = 100, offset: int = 0) -> list:
    response = requests.get('https://api.vk.com/method/wall.get', params={
        'access_token': os.getenv('ACCESS_TOKEN'),
        'domain': domain,
        'count': count,
        'offset': offset,
        'v': '5.199'
    })
    return response.json()['response']['items']

def parse_wall(domain: str, count: int):
    responses = []
    for i in range(1 if count < 100 else count // 100):
        offset = i * 100
        try:
            response = get_vk_posts(domain, count=100, offset=offset)
        except Exception as e:
            print(f"Error fetching posts, stopping: {e}")
            break
        if len(response) == 0:
            break
        responses.append(response)
    
    return responses

def extract_texts(responses: list) -> list:
    texts = []
    for response in responses:
        for item in response:
            if 'text' in item and len(item['attachments']) == 0:
                text = item['text']
                # Remove HTML tags
                text = re.sub(r'<[^>]+>', '', text)
                # Remove links
                text = re.sub(r'http\S+', '', text)
                texts.append(text)

    return texts

def parse_web(url: str, pages_limit=2000, curr_page=0) -> list:
    # if curr_page >= pages_limit:
    #     return []
    # response = requests.get(url).text
    # soup = BeautifulSoup(response, 'html.parser')
    # texts = []
    # for item in soup.find_all('div', class_='topicbox'):
    #     try:
    #         text = item.find('div', class_='text').get_text()

    #         # Remove HTML tags
    #         text = re.sub(r'<[^>]+>', '', text)
    #         # Remove links
    #         text = re.sub(r'http\S+', '', text)
    #         texts.append(text)
    #     except:
    #         continue

    # next_button = soup.find('div', class_='voteresult')
        
    # next_button = next_button.find('a').attrs
    # next_url = next_button['href']
    # if next_url:
    #     texts += parse_web('https://anekdot.ru' + next_url, pages_limit=pages_limit, curr_page=curr_page + 1)
    # return texts

    texts = []
    for i in range(pages_limit):
        response = requests.get(url).text
        soup = BeautifulSoup(response, 'html.parser')
        for item in soup.find_all('div', class_='topicbox'):
            try:
                text = item.find('div', class_='text').get_text()

                # Remove HTML tags
                text = re.sub(r'<[^>]+>', '', text)
                # Remove links
                text = re.sub(r'http\S+', '', text)
                texts.append(text)
            except:
                continue

        buttons_div = soup.find('div', class_='voteresult')
        
        buttons = buttons_div.find_all('a')
        # next_url = next_button['href']
        if any('Вчера' in button.text for button in buttons):
            next_button = [button for button in buttons if 'Вчера' in button.text][0]
            url = 'https://anekdot.ru' + next_button['href']
        else:
            print("No more pages to scrape.")
            break

    return texts

def save_to_csv(texts: list, filename: str):
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Text'])
        for text in texts:
            writer.writerow([text])

if __name__ == "__main__":
    # load_dotenv()
    # domains = ['jumoreski', 'anekdotikategoriib']
    # count = 200000
    
    # for domain in domains:
    #     response = parse_wall(domain, count)
    #     texts = extract_texts(response)
    #     save_to_csv(texts, f'data/output_{domain}.csv')
    #     print(f"{len(texts)} saved to output_{domain}.csv")

    url='https://www.anekdot.ru/release/anekdot/day/2025-05-09/'
    texts = parse_web(url, pages_limit=5000)
    save_to_csv(texts, 'data/output_anekdot_web.csv')
    print(f"{len(texts)} saved to output_anekdot_web.csv")