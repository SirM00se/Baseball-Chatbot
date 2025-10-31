import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from dataScraper import get_rule_links, fetch_info

@pytest.fixture
def mock_driver():
    """Fixture that creates a fake Selenium driver with mock links."""
    driver = MagicMock()
    mock_element1 = MagicMock()
    mock_element1.get_attribute.return_value = "https://mlb.com/glossary/rules/test1"
    mock_element2 = MagicMock()
    mock_element2.get_attribute.return_value = "https://mlb.com/glossary/rules/test2"
    driver.find_elements.return_value = [mock_element1, mock_element2]
    return driver


@patch("dataScraper.requests.get")
def test_dataScraper_integration(mock_get, mock_driver, tmp_path):
    """
    Integration test:
    - Simulate Selenium collecting rule links.
    - Simulate BeautifulSoup parsing HTML for each page.
    - Combine results into a DataFrame.
    """

    # Step 1: mock HTML returned by requests.get()
    fake_html = b"""
    <html><body>
        <h5>Term</h5>
        <p>Description of term.</p>
        <p><strong>Note:</strong> extra info.</p>
    </body></html>
    """
    mock_get.return_value.content = fake_html

    # Step 2: get all URLs (from mock Selenium)
    urls = get_rule_links(mock_driver, "https://mlb.com/glossary/rules")
    assert len(urls) == 2  # we should get two URLs

    # Step 3: scrape each URL
    all_data = []
    for url in urls:
        all_data.extend(fetch_info(url))

    # Step 4: build DataFrame and save to temporary CSV
    df = pd.DataFrame(all_data, columns=["url", "text", "tag"])
    output_path = tmp_path / "output.csv"
    df.to_csv(output_path, index=False)

    # Step 5: verify everything worked together
    assert output_path.exists()
    assert len(df) > 0
    assert "Description of term" in df["text"].iloc[0]
    assert all(df["url"].str.startswith("https://mlb.com"))

    print("\nIntegration test passed")
