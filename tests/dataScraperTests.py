import pytest
from unittest.mock import MagicMock, patch
from dataScraper import create_driver, get_rule_links, fetch_info

def test_create_driver_success():
    with patch("dataScraper.webdriver.Chrome") as mock_chrome:
        driver = MagicMock()
        mock_chrome.return_value = driver
        result = create_driver()
        assert result == driver

def test_get_rule_links_parses_links():
    mock_driver = MagicMock()
    mock_element = MagicMock()
    mock_element.get_attribute.return_value = "https://example.com/rule1"
    mock_driver.find_elements.return_value = [mock_element]

    urls = get_rule_links(mock_driver, "https://fake.com")
    assert urls == ["https://example.com/rule1"]

@patch("dataScraper.requests.get")
def test_fetch_info_parses_html(mock_get):
    html = """
    <html><body>
        <p>First paragraph.</p>
        <h5>Header</h5>
        <p><strong>Rule:</strong> Some description.</p>
    </body></html>
    """
    mock_get.return_value.content = html.encode()
    result = fetch_info("https://www.mlb.com/glossary/rules/sample")
    assert any("First paragraph" in r[1] for r in result)
