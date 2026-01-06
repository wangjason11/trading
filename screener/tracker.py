### SQL Create Stock Calendar -- DB Browser
#CREATE TABLE calendar
#(
#	date DATE
#	,holiday TEXT
#	,day_count BIGINT
#);

### Import Calendar CSV -- CMD
#sqlite3 "C:\Users\Jason\Documents\codingproj\Stock Screens\stocks.sqlite3"
# .separator ","
# .import 'C:\Users\Jason\Documents\codingproj\Stock Screens\market_calendar.csv' calendar

### Task Scheduler
#Program: C:\Windows\System32\cmd.exe
#/C python tracker_vfinal.py
#Start-in: C:\Users\Jason\Documents\codingproj\Stock Screens\

### Turn off AUC
### Allow Wake Timers: enabled 

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import sys 
import time
from bs4 import BeautifulSoup
import datetime 
# import csv
import sqlite3
import requests
from yahoo_finance import Share
from googlefinance import getQuotes
import json 

### Login and retrieve screen url

conn = sqlite3.connect('stocks.sqlite3')
c = conn.cursor()

c.execute('SELECT holiday FROM calendar WHERE date = ?', (datetime.datetime.today().strftime('%Y-%m-%d'),))
market_close = c.fetchone()
if market_close[0] == 'TRUE':
	sys.exit()
else:
	urls = dict()
	urls['rekking'] = 'https://research2.fidelity.com/fidelity/screeners/commonstock/main.asp?saved=5ec8b3c0934949e682a56020f9f0c956'
	urls['boosting'] = 'https://research2.fidelity.com/fidelity/screeners/commonstock/main.asp?saved=f5251c11f1cd42da911ec101628dd978'

	screen_res = list()

	driver = webdriver.Chrome('C:\Anaconda2\selenium\webdriver\chromedriver.exe')
	driver.get('https://login.fidelity.com/ftgw/Fidelity/RtlCust/Login/Init/df.chf.ra/trial')
	
	username = driver.find_element_by_name('SSN-visible')
	password = driver.find_element_by_name('PIN')
	
	username.send_keys('u13e12')
	password.send_keys('13371337')
	driver.find_element_by_xpath('//*[@id="Login"]/ol/li[4]/button').click()

time.sleep(5)

for key in urls:
	driver.get(urls[key])
	try:
		wait = WebDriverWait(driver, 10)
		element = wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="WSOD"]/div[3]/div/div[6]/a')))
	except:
		wait = WebDriverWait(driver, 25)
		element = wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="WSOD"]/div[3]/div/div[6]/a')))
		
### Parse html with BeautifulSoup

	html = driver.page_source
	soup = BeautifulSoup(html, 'html.parser')

	tags = soup('td')

	for tag in tags:
		if tag.get('raw', None) == None : continue 
		if tag['class'] == [u'col-checkbox'] and tag.get('raw', None) == u'':
			screen_res.append(datetime.datetime.today().strftime('%Y-%m-%d'))
			screen_res.append(key)
		elif tag['class'] == [u'content-text'] and tag.get('raw', None) == u'':
			screen_res.append(tag.string)
		else:
			screen_res.append(tag.get('raw', None))
		
stock_rows = [screen_res[i:i + 22] for i in xrange(0, len(screen_res), 22)]

driver.quit()

### Creating Database Tables 

c.execute('''CREATE TABLE IF NOT EXISTS screens 
	(day_bought DATE
	,portfolio TEXT
	,score TINYINT
	,name TEXT
	,ticker VARCHAR(6)
	,type TEXT
	,perf_today DECIMAL(4,2)
	,perf_5days DECIMAL(4,2)
	,perf_4wk DECIMAL(4,2)
	,perf_13wk DECIMAL(4,2)
	,volume REAL
	,market_cap REAL
	,volume_today_10davg REAL
	,off_10d_SMA DECIMAL(4,2)
	,off_20d_SMA DECIMAL(4,2)
	,off_50d_SMA DECIMAL(4,2)
	,off_200d_SMA DECIMAL(4,2)
	,change_52wkhi DECIMAL(4,2)
	,change_52wklo DECIMAL(4,2)
	,exchange VARCHAR(7)
	,price REAL
	,industry TEXT)''')

c.execute('''CREATE TABLE IF NOT EXISTS prices 
	(ticker VARCHAR(6)
	,portfolio TEXT
	,day_bought DATE
	,date DATE
	,price REAL)''')

c.execute('''CREATE TABLE IF NOT EXISTS tracker 
	(ticker VARCHAR(6)
	,portfolio TEXT
	,day_bought DATE
	,buy_price REAL
	,shares BIGINT
	,cost REAL
	,today DATE
	,price_today REAL
	,value REAL
	,pct_change DECIMAL(4,2)
	,days_elapsed BIGINT
	,sold BOOLEAN
	,nom_gains REAL)''')
conn.commit()

for stock in stock_rows:
	c.execute('INSERT INTO screens VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', stock)
conn.commit()

### Retrieve Stock Price Quotes 

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

fidelity_quote = 'http://quotes.fidelity.com/webxpress/get_quote?QUOTE_TYPE=R&SID_VALUE_ID='

for row in data:
	stock_url = fidelity_quote + row[0]
	stock_page = requests.get(stock_url).content
	soup = BeautifulSoup(stock_page, 'html.parser')
	c.execute('INSERT INTO prices VALUES (?,?,?,?,?)',
	(row[0], row[1], row[2], datetime.datetime.today().strftime('%Y-%m-%d'), soup.find('td', attrs={'class': 'SmallDataHeader'}).text.strip()[:-5]))
conn.commit()

### Populating Tracker Table

c.execute('''INSERT INTO tracker 
	SELECT a.ticker ticker
	,a.portfolio portfolio
	,a.day_bought day_bought
	,a.price buy_price
	,cast(10000/a.price as int) AS shares
	,cast(10000/a.price as int)*a.price AS cost
	,b.date today
	,b.price price_today
	,cast(10000/a.price as int)*b.price AS value
	,(b.price/a.price - 1) AS pct_change
	,d.day_count - c.day_count AS days_elapsed
	,CASE WHEN 
		(b.price/a.price - 1)*100 >= 10 OR
		d.day_count - c.day_count >= 30 THEN 1
		ELSE 0 END AS sold
	,CASE WHEN
		(b.price/a.price - 1)*100 >= 10 OR d.day_count - c.day_count >= 30 
		THEN cast(10000/a.price as int)*b.price - cast(10000/a.price as int)*a.price
		ELSE 0 END AS nom_gains
	FROM screens a
	LEFT JOIN prices b
	ON a.ticker = b.ticker
	AND a.portfolio = b.portfolio
	AND a.day_bought = b.day_bought
	LEFT JOIN calendar c
	ON a.day_bought = c.date
	LEFT JOIN calendar d
	ON b.date = d.date
	WHERE b.date = strftime('%Y-%m-%d', 'now', 'localtime')
	ORDER BY a.day_bought asc, a.portfolio asc, a.score desc''')
conn.commit()

### Tracking Portfolio Performance

c.execute('''CREATE TABLE IF NOT EXISTS portfolio
	(open_date DATE
	,portfolio TEXT
	,today DATE
	,days_elapsed BIGINT
	,stocks_bought BIGINT
	,alltime_value REAL
	,stocks_held BIGINT
	,portf_value REAL
	,nom_cost REAL
	,real_cost REAL
	,nom_gains REAL
	,portf_return DECIMAL(4,2)
)''')
	
c.execute('''INSERT INTO portfolio 
	SELECT
	d.open_date open_date
	,a.portfolio portfolio
	,strftime('%Y-%m-%d', 'now', 'localtime') AS today
	,MAX(c.days_elapsed) AS days_elapsed
	,COUNT(a.ticker) AS stocks_bought
	,SUM(c.value) AS alltime_value
	,COUNT(CASE WHEN c.sold = 1 THEN 0 else a.ticker END) stocks_held
	,SUM(CASE WHEN c.sold = 1 THEN 0 ELSE c.value END) AS portf_value
	,SUM(c.cost) nom_cost
	,SUM(c.cost) - SUM(CASE WHEN c.sold = 1 THEN c.value ELSE 0 END) AS real_cost
	,SUM(c.nom_gains) AS nom_gains
	,SUM(CASE WHEN c.sold = 1 THEN 0 ELSE c.value END)/
		(SUM(c.cost) - SUM(CASE WHEN c.sold = 1 THEN c.value ELSE 0 END)) - 1 AS portf_return
	FROM screens a
	LEFT JOIN (SELECT ticker
		,portfolio
		,day_bought
		,MAX(days_elapsed) max_day FROM tracker
		GROUP BY 
			ticker
			,portfolio
			,day_bought) b
	ON a.ticker = b.ticker
	AND a.day_bought = b.day_bought
	AND a.portfolio = b.portfolio
	LEFT JOIN tracker c 
	ON b.ticker = c.ticker
	AND b.day_bought = c.day_bought
	AND b.portfolio = c.portfolio
	AND b.max_day = c.days_elapsed
	LEFT JOIN (
		SELECT portfolio, MIN(day_bought) AS open_date FROM screens GROUP BY portfolio) d
	ON a.portfolio = d.portfolio
	GROUP BY 
		d.open_date 
		,a.portfolio
		,strftime('%Y-%m-%d', 'now', 'localtime') 
	ORDER BY 
		d.open_date asc, a.portfolio asc''')
conn.commit()
