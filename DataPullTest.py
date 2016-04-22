'''
Created on Mar 16, 2016

@author: Alex
'''
import numpy as np
import scipy as sp
import matplotlib
import csv as csv
import urllib2
from sklearn.svm import LinearSVC
from bs4 import BeautifulSoup, SoupStrainer
import requests
from requests.auth import HTTPBasicAuth
import re
from sklearn import decomposition
from sklearn import datasets
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import datetime
from sympy.functions.elementary.complexes import principal_branch

"""r = requests.get('http://api.sportsdatabase.com/nfl/query.json?sdql=date%2Cpoints%40team%3DBears%20and%20season%3D2011&output=json&api_key=guest')
print(r.status_code)
print(r.text)"""


"""player_page_links = []
for i in range(0, 26):
    response = urllib2.urlopen('http://www.basketball-reference.com/players/' + chr(i + 97))
    soup = BeautifulSoup(response.read(), 'html.parser', parseOnlyThese=SoupStrainer('strong'))
    for link in soup.find_all('a'):
        player_page_links.append(str('http://www.basketball-reference.com' + link['href']))
    print(player_page_links)
"""

def scrape_rosters():
    """Scrapes the rosters of every team for individual player stats, returns
    dict of numpy arrays"""
    print "got here"
    all_rosters = {}
    franchise_codes = ['ATL', 'BOS', 'BRK', 'CHO', 'CHI', 'CLE', 'DAL', 'DEN',
                       'DET', 'GSW', 'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA',
                       'MIL', 'MIN', 'NOH', 'NYK', 'OKC', 'ORL', 'PHI', 'PHO',
                       'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS']
    for code in franchise_codes:
        print "got here"
        url = 'http://www.basketball-reference.com/teams/' + code +\
            '/2015.html#totals::none'
        response = urllib2.urlopen(url)
        soup = BeautifulSoup(response.read(), 'html.parser',
                             parse_only=SoupStrainer(id='all_totals'))
        all_stats = []
        for td in soup.findAll('td'):
            all_stats.append(td.string)
        partitioned = []
        for i in range(0, len(all_stats), 28):
            partitioned.append(all_stats[i:i+28])
        roster_array = np.array(partitioned)
        all_rosters[code] = roster_array
    print all_rosters
    return all_rosters
    pass


def scrape_rivalry_history(team_code, opponent_code):
    """Scrapes history of team v. team. Results returned as numpy 2D array with date 
    and pt difference. 
    E.g. BOS wins by 7 pts against NOH on 4/3/16: [[u'Sun, Apr 3, 2016' u'+7']...]"""
    url = 'http://www.basketball-reference.com/play-index/rivals.cgi?request=1'\
        + '&team_id=' + team_code + '&opp_id=' + opponent_code +'&is_playoffs='
    response = urllib2.urlopen(url)
    soup = BeautifulSoup(response.read(), 'html.parser',
                         parse_only=SoupStrainer('tbody'))
    raw_strings = []
    for td in soup.findAll('td'):
        raw_strings.append(td.string)
    date_diff_tuples = []
    for i in range(1, len(raw_strings), 16):
        date_diff_tuples.append((raw_strings[i], raw_strings[i + 10]))
    return np.array(date_diff_tuples)


def scrape_data(until_this_date):
    franchise_codes = ['ATL', 'BOS', 'BRK', 'CHO', 'CHI', 'CLE', 'DAL', 'DEN',
                       'DET', 'GSW', 'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA',
                       'MIL', 'MIN', 'NOP', 'NYK', 'OKC', 'ORL', 'PHI', 'PHO',
                       'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS']
    franchise_numbers = [float(i) for i in xrange(31)]
    # create dictionary of franchise codes to numbers
    code_to_number = dict(zip(franchise_codes, franchise_numbers))
    list_of_game_stats = []
    target = []
    url_base = "http://www.basketball-reference.com/play-index/tgl_finder.cgi?request=1&match=game&lg_id=NBA&year_min=2016&year_max=2016&team_id=&opp_id=&is_playoffs=N&round_id=&best_of=&team_seed_cmp=eq&team_seed=&opp_seed_cmp=eq&opp_seed=&is_range=N&game_num_type=team&game_num_min=&game_num_max=&game_month=&game_location=&game_result=&is_overtime=&c1stat=pts&c1comp=gt&c1val=&c2stat=ast&c2comp=gt&c2val=&c3stat=drb&c3comp=gt&c3val=&c4stat=ts_pct&c4comp=gt&c4val=&c5stat=&c5comp=gt&c5val=&order_by=date_game&order_by_asc=Y&offset="
    offsets = [i * 100 for i in xrange(25)]
    date = '' 
    j = 0
    for offset in offsets:
        j += 1
        print "scraping ", j, "th page of stats"
        url = url_base + str(offset)
        r = requests.get(url)
        soup = BeautifulSoup(r.text, "html.parser")
        games = soup.findAll("tr", class_=[u''])[2:]
        break_from_outer_loop = False
        for game in games:
    #         For some reason <tr class=" thead"> satisfies class_=[u''] even though 
    #         "" != " thead" so I filter those out here. If you can figure out a fix
    #         so that those classes don't get picked up in games then please implement.
            if game["class"] == [u'', u'thead']:
                continue
            raw_game_stats = game.find_all("td")
            # get the date
            date = raw_game_stats[1].get_text()
            date = datetime.datetime(int(date[0:4]), int(date[5:7]), int(date[8:]))
            # don't collect this data if we've passed the until date and break loop
            if date >= until_this_date:
                break_from_outer_loop = True
                break;
            # convert the franchise codes to numbers
            team1_id = code_to_number[raw_game_stats[2].get_text()]
            team2_id = code_to_number[raw_game_stats[4].get_text()]
            # create a single row of the matrix with stats for one game
            game_stats = [team1_id] + [team2_id] + \
                [float(raw_game_stats[i].get_text()) for i in xrange(6, 61)]
            # add row to list of rows to be made into numpy array
            list_of_game_stats.append(game_stats)
            # set binary vector target data by comparing points scored
            outcome = 1 if game_stats[15] > game_stats[42] else 0
            target.append(outcome)
        if break_from_outer_loop:
            break;      
    target = np.array(target)
    data = np.array(list_of_game_stats)
    print data.shape
    print target.shape 
    print data
    return data, target
    
# do PCA stuff
# This reduces the number of columns in the matrix but also changes the numbers.
# I have no idea what the numbers mean or which columns were kept.    
def PCA(data):
    pca = decomposition.PCA(n_components=20)
    princinpal_component_data = pca.fit_transform(data)
    print(princinpal_component_data.shape)
    return princinpal_component_data

def classify(data, target, test_data):

    
    #KNeighbors Classification
    #Split 20 = .80 F1-Score
    #Split 500 = .94 F1-Score
    #model = KNeighborsClassifier(n_neighbors=19)
    #model.fit(data, target)
   
    #DecisionTree Classifier
    #Split 20 = .65 F1-Score
    #Split 500 = 0.92 F1-Score
    #model = DecisionTreeClassifier(max_depth=10)
    #model.fit(data, target)
    
    
    #Linear SVC
    #Split 20 = .91 F1-Score
    #Split 500 = .99 F1-Score
    model = LinearSVC()
    model.fit(data, target)

    
    # make predictions
    predicted = model.predict(test_data)
    return predicted

def analyze(data, target):
    '''
    Analyze the accuracy of your classifier against the validation dataset.
    Print out the table of its precision, recall and f1-score.

    Try this out with the validation dataset provided.
    '''
    splitData = 500
    training_data, test_data, training_target, test_target = data[:splitData], data[splitData:], target[:splitData], target[splitData:]
    #print(training_data.shape, test_data.shape, training_target.shape, test_target.shape)
    classified = classify(training_data, training_target, test_data)
    print(classified, test_target)
    #print(metrics.confusion_matrix(classified, test_target))
    print(metrics.classification_report(classified, test_target))



def main():
    # all data from games up until this date is processed
    until_this_date = datetime.datetime(2016, 01, 20)
    data, target = scrape_data(until_this_date)
    #data_to_train_classifier_on = PCA(data)
    analyze(data, target)
