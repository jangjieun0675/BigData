# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 04:14:49 2024

@author: ATIV
"""

import csv
import matplotlib.pyplot as plt

with open('중국_방한외래관광객_2018_202412.csv') as f:
    data = csv.reader(f)
    next(data)
    result = []
    months = []
    values = []
    for row in data:
        months.append(row[2])
        values.append(int(row[3]))

plt.figure(figsize = (10,5), dpi=100)
plt.rc('font', family='Malgun Gothic')
plt.title('Visitors')
plt.bar(range(len(values)), values, color='b')
plt.xlabel('yyyymm')
plt.ylabel('Person')

plt.xticks(range(len(months)), months, rotation=90)
plt.tight_layout()
plt.show()





from bs4 import BeautifulSoup
import pandas as pd

file = open("weather.html", encoding='utf-8').read()
soup = BeautifulSoup(file, 'html.parser')

locations = []
temperatures = []
humidities = []

tag_tbody = soup.find('tbody')
for row in tag_tbody.find_all('tr'):
    cell = row.find('th')
    location = cell.text.strip() if cell else None
    cells = row.find_all('td')
    temperature = cells[4].text.strip() if len(cells) >= 6 else None
    humidity = cells[9].text.strip().replace('%', '') if len(cells) >= 11 else None
    if location and temperature and humidity:
        locations.append(location)
        temperatures.append(temperature)
        humidities.append(humidity)

weather_df = pd.DataFrame({
    'sido-gu': locations,
    '온도': temperatures,
    '습도': humidities
})

print(weather_df)




    
    
    
    