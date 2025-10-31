import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

import dataScraper  # your main script


@patch("dataScraper.requests.get")
@patch("dataScraper.webdriver.Chrome")
@patch("dataScraper.ChromeDriverManager.install")
def test_end_to_end_dataScraper(mock_install, mock_chrome, mock_get, tmp_path):
    """
    E2E test for the dataScraper:
    - Mocks Selenium and requests
    - Runs the real `main()` function
    - Verifies the CSV output
    """

    # --- Step 1: Setup ChromeDriver mock ---
    mock_install.return_value = "fake/driver/path"
    driver = MagicMock()
    mock_chrome.return_value = driver

    # Fake Selenium link elements
    mock_element1 = MagicMock()
    mock_element1.get_attribute.return_value = "https://mlb.com/glossary/rules/test1"
    mock_element2 = MagicMock()
    mock_element2.get_attribute.return_value = "https://mlb.com/glossary/rules/test2"
    driver.find_elements.return_value = [mock_element1, mock_element2]

    # --- Step 2: Fake HTML page responses ---
    html_page = b"""
    <html>
      <body>
        <h5>Definition</h5>
        <p>This is the rule description.</p>
        <p><strong>Example:</strong> extra details.</p>
      </body>
    </html>
    """
    mock_get.return_value.content = html_page

    # --- Step 3: Patch output path to a temporary directory ---
    output_path = tmp_path / "baseballrules.csv"

    # Monkey-patch main()’s CSV save path
    with patch.object(dataScraper, "main") as mock_main:
        def fake_main():
            base_url = "https://mlb.com/glossary/rules"
            driver = dataScraper.create_driver(headless=True)
            urls = dataScraper.get_rule_links(driver, base_url)
            driver.quit()

            all_data = []
            for url in urls:
                all_data.extend(dataScraper.fetch_info(url))

            df = pd.DataFrame(all_data, columns=["url", "text", "tag"])
            df.to_csv(output_path, index=False)
            print(f"Data saved to: {output_path}")

        mock_main.side_effect = fake_main

        # --- Step 4: Run the real pipeline ---
        dataScraper.main()  # triggers our fake_main

    # --- Step 5: Verify output ---
    assert output_path.exists(), "CSV file was not created"

    df = pd.read_csv(output_path)
    assert len(df) > 0
    assert "url" in df.columns
    assert "text" in df.columns
    assert any("rule description" in text.lower() for text in df["text"])

    print("\n End-to-End test passed — full pipeline works!")
