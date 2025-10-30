import requests
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common import NoSuchElementException
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
    try:
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=chrome_options
        )
        return driver
    except Exception as e:
        print(f"Failed to start Chrome Driver: {e}")
        return None


# -----------------------------
# Collect all rule page URLs
# -----------------------------
def get_rule_links(driver, base_url):
    urls = []
    try:
        driver.get(base_url)
        WebDriverWait(driver, 10)
        try:
            links = driver.find_elements(By.XPATH, "//a[contains(@class, 'p-related-links__link')]")
            if links is not None:
                urls = [link.get_attribute("href") for link in links if link.get_attribute("href")]
            print(f"Found {len(urls)} links on this page")
        except NoSuchElementException:
            print("Inner error: Element not found, skipping this part")
    except Exception as e:
        print(f"Failed to get links from {base_url}: {e}")
    return urls


# -----------------------------
# Extract text from one rule page
# -----------------------------
def fetch_info(url):
    """Fetchs all paragraph text from one MLB glossary page."""
    data = []
    tag = url[35:]
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        data = []  # collect rows as [url, text, tag]
        paragraphs = soup.find_all(["p", "h5"])
        if not paragraphs:
            print("No paragraphs found")
        strong = ""

        for paragraph in paragraphs:
            for a in paragraph.find_all("a"):
                a.decompose()

            if paragraph.name == 'h5':
                strong = paragraph.text
            elif paragraph.find('strong'):
                strong = paragraph.get_text(strip=True)
                if not strong:
                    print("No text found")
            else:
                clean_text = paragraph.get_text(strip=True)
                if clean_text:
                    if strong == "":
                        data.append([url, clean_text, tag])
                    else:
                        data.append([url, f"{strong}: {clean_text}", tag])
                        strong = ""
        print(f"Finished scraping: {url}")
    except requests.RequestException as e:
        print(f"Request error for {url}: {e}")
    except Exception as e:
        print(f"Unexpected error for {url}: {e}")
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