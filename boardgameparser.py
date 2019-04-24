import requests
from bs4 import BeautifulSoup
import pandas as pd
import urllib.request
import time
import datetime

def creepy_crawly(first_page, last_page):
    index = first_page

    df = pd.DataFrame()
    while index <= last_page:

        url = "https://boardgamegeek.com/browse/boardgame/index/" + str(index)
        source_code = requests.get(url)
        source_text = source_code.text
        soup = BeautifulSoup(source_text, "html.parser")
        
        for i in soup.findAll('tr', {'id' : 'row_'}):
            soup_i = BeautifulSoup(str(i), "html.parser")
            #Temp_rank = soup_i.find('td', {'class': 'collection_rank'})
            #rank = Temp_rank.find('a').get("name")
            Temp_name = soup_i.find('td', {'class': 'collection_objectname'})
            Soup_name = BeautifulSoup(str(Temp_name), "html.parser")
            name = Soup_name.find('a').get_text()
            Temp_rating = soup_i.findAll('td', {'class': 'collection_bggrating'})
            rating = []

            for j in Temp_rating:
                rating.append(j.get_text())
            
            Geek_Rating = rating[0]
            Avg_Rating = rating[1]
            Num_Voters = rating[2]

            GeekRating = Geek_Rating.strip()
            AvgRating = Avg_Rating.strip()
            NumVoters = Num_Voters.strip()

            df = df.append({
            'Rank': rank,
            'Name':name,
            'GeekRating': GeekRating,
            'AverageRating' :AvgRating,
            'NumberOfVoters' :NumVoters
            }, ignore_index=True)

        index += 1

        df.to_csv("parsed_results/boardgamegeekdat.csv")

creepy_crawly(1,1066)