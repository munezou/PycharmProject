import requests
from bs4 import BeautifulSoup

xml = requests.get('http://news.yahoo.co.jp/pickup/science/rss.xml')
soup = BeautifulSoup(xml.text, 'html.parser')
for news in soup.findAll('item'):
    print(news.title.string)
