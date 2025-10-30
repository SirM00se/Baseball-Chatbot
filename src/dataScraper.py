import requests
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait


# -----------------------------
# Setup Selenium
# -----------------------------
def create_driver(headless=True):
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless")
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    Service(executable_path='../chromedriver.exe')
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=chrome_options
    )
    return driver


# -----------------------------
# Collect all rule page URLs
# -----------------------------
def get_rule_links(driver, base_url):
    driver.get(base_url)
    WebDriverWait(driver, 10)
    links = driver.find_elements(By.XPATH, "//a[contains(@class, 'p-related-links__link')]")
    urls = [link.get_attribute("href") for link in links if link.get_attribute("href")]
    return urls


# -----------------------------
# Extract text from one rule page
# -----------------------------
def fetch_info(url):
    """Fetchs all paragraph text from one MLB glossary page."""
    tag = url[35:]
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    data = []  # collect rows as [url, text, tag]
    paragraphs = soup.find_all(["p", "h5"])
    strong = ""

    for paragraph in paragraphs:
        for a in paragraph.find_all("a"):
            a.decompose()

        if paragraph.name == 'h5':
            strong = paragraph.text
        elif paragraph.find('strong'):
            strong = paragraph.get_text(strip=True)
        else:
            clean_text = paragraph.get_text(strip=True)
            if clean_text:
                if strong == "":
                    data.append([url, clean_text, tag])
                else:
                    data.append([url, f"{strong}: {clean_text}", tag])
                    strong = ""
    print(f"Finished scraping: {url}")
    return data


# -----------------------------
# Main control function
# -----------------------------
def main():
    base_url = "https://www.mlb.com/glossary/rules"
    driver = create_driver(headless=True)

    # Collect all rule page URLs
    urls = get_rule_links(driver, base_url)
    driver.quit()

    # Collect data from each page
    all_data = []
    for url in urls:
        try:
            all_data.extend(fetch_info(url))
        except Exception as e:
            print(f"Failed to scrape {url}: {e}")

    # Build DataFrame
    df = pd.DataFrame(all_data, columns=["url", "text", "tag"])

    # Save to CSV
    output_path = "../data/baseballrules.csv"
    df.to_csv(output_path, index=False)
    print(f"\nData saved to: {output_path}")


# -----------------------------
# Run script
# -----------------------------
if __name__ == "__main__":
    main()