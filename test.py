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
def fetchinfo(url):#collects text data
    tag = url[35:]#collects tag
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    paragraphs = soup.find_all(["p","h5"])
    strong = ""
    for paragraph in paragraphs:
        for a in paragraph.find_all("a"):#checks if paragraph contains a hyperlink and removes it
            a.decompose()
        if paragraph.name == 'h5':
            strong = paragraph.text
        elif paragraph.find('strong'):
            strong = paragraph.get_text(strip=True)
        else:
            clean_text = paragraph.get_text(strip=True)
            if clean_text:#adds text to csv file
                if strong == "":
                    df.loc[len(df)] = [url, clean_text, tag]
                else:
                    df.loc[len(df)] = [url, strong+": "+clean_text, tag]
                    strong = ""
    print("page complete")
service = Service(executable_path='chromedriver.exe')
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
df = pd.DataFrame(columns=["url", "text", "tag"])
baseurl = f'https://www.mlb.com/glossary/rules'
urls = []
driver.get(baseurl)
wait = WebDriverWait(driver, 10)
link_elements = driver.find_elements(By.XPATH, "//a[contains(@class, 'p-related-links__link')]")#finds all links on basepage
for link in link_elements:#appends href to url
    href = link.get_attribute("href")
    urls.append(href)
for url in urls:
    fetchinfo(url)
df.to_csv('baseballrules.csv', index=False)