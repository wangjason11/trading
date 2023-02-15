from bs4 import BeautifulSoup
import requests
import sqlite3

conn = sqlite3.connect('stocks.sqlite3')
c = conn.cursor()

c.execute('''SELECT a.ticker, a.portfolio, a.day_bought
	FROM screens a
	LEFT JOIN (
		SELECT ticker
			,portfolio
			,day_bought
			,MAX(days_elapsed) max_day 
		FROM tracker GROUP BY 
			ticker
			,portfolio
			,day_bought) b
	ON a.ticker = b.ticker
	AND a.portfolio = b.portfolio
	AND a.day_bought = b.day_bought
	LEFT JOIN tracker c 
	ON b.ticker = c.ticker
	AND b.day_bought = c.day_bought
	AND b.portfolio = c.portfolio
	AND b.max_day = c.days_elapsed
	WHERE (c.sold <> 1 OR c.sold IS NULL)
	ORDER BY a.day_bought asc, a.portfolio asc, a.score desc
''')

data = c.fetchall()

yahoo_url = 'http://finance.yahoo.com/quote/'

for row in data:
	try:
### Yahoo Finance BeautifulSoup Parse
		stock_url = yahoo_url + row[0]
		stock_page = requests.get(stock_url).content
		soup = BeautifulSoup(stock_page, 'html.parser')
#		print soup.find('span', attrs={'class': 'Fw(b) Fz(36px) Mb(-4px)'}).text
		c.execute('INSERT INTO prices VALUES (?,?,?,?,?)', 
		(row[0], row[1], row[2], datetime.datetime.today().strftime('%m/%d/%Y'), soup.find('span', attrs={'class': 'Fw(b) Fz(36px) Mb(-4px)'}).text))
	except:
		print row[0]
### Yahoo Finance Python Package
#		print row
#		c.execute('INSERT INTO prices VALUES (?,?,?,?,?)', 
#		(row[0], row[1], row[2], datetime.datetime.today().strftime('%m/%d/%Y'), Share(row[0]).get_price()))