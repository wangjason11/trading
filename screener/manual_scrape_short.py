from bs4 import BeautifulSoup
import requests

fidelity_quote = 'http://quotes.fidelity.com/webxpress/get_quote?QUOTE_TYPE=R&SID_VALUE_ID='

### Fidelity Quote Search
stock_url = fidelity_quote + 'aapl'
stock_page = requests.get(stock_url).content
soup = BeautifulSoup(stock_page, 'html.parser')
quote = soup.find('td', attrs={'class': 'SmallDataHeader'}).text.strip()[:-5]
print quote


