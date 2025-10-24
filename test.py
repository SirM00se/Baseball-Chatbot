import requests
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
def fetchinfo(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    paragraphs = soup.find_all("p")
    strong = ""
    for paragraph in paragraphs:
        for a in paragraph.find_all("a"):
            a.decompose()
        if paragraph.find('strong'):
            strong = paragraph.get_text(strip=True)
        else:
            clean_text = paragraph.get_text(strip=True)
            if clean_text:
                if strong == "":
                    df.loc[len(df)] = [url, clean_text]
                else:
                    df.loc[len(df)] = [url, strong+": "+clean_text]
                    strong = ""
    print("page complete")
service = Service(executable_path='chromedriver.exe')
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
df = pd.DataFrame(columns=["url", "text"])
baseurl = f'https://www.mlb.com/glossary/rules'
urls = []
driver.get(baseurl)
wait = WebDriverWait(driver, 10)
link_elements = driver.find_elements(By.XPATH, "//a[contains(@class, 'p-related-links__link')]")
for link in link_elements:
    href = link.get_attribute("href")
    urls.append(href)
for url in urls:
    fetchinfo(url)
df.to_csv('baseballrules.csv', index=False)