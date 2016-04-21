
'''
Created on Mar 16, 2016
'''
import numpy as np
import urllib2
from bs4 import BeautifulSoup, SoupStrainer
import requests
from sklearn import decomposition
from sklearn import datasets, naive_bayes, metrics
import datetime


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
    pass


def create_matrix(from_this_date, until_this_date, season):
    franchise_codes = ['ATL', 'BOS', 'BRK', 'CHO', 'CHI', 'CLE', 'DAL', 'DEN',
                       'DET', 'GSW', 'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA',
                       'MIL', 'MIN', 'NOP', 'NYK', 'OKC', 'ORL', 'PHI', 'PHO',
                       'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS']
    franchise_numbers = [float(i) for i in xrange(31)]
    # create dictionary of franchise codes to numbers
    code_to_number = dict(zip(franchise_codes, franchise_numbers))
    list_of_game_stats = []
    target = []
    url_base = "http://www.basketball-reference.com/play-index/tgl_finder.cgi?request=1&match=game&lg_id=NBA&year_min=" +\
    str(season) + "&year_max=" + str(season) + "&team_id=&opp_id=&is_playoffs=N&round_id=&best_of=&team_seed_cmp=eq&team_seed=&opp_seed_cmp=eq&opp_seed=&is_range=N&game_num_type=team&game_num_min=&game_num_max=&game_month=&game_location=&game_result=&is_overtime=&c1stat=pts&c1comp=gt&c1val=&c2stat=ast&c2comp=gt&c2val=&c3stat=drb&c3comp=gt&c3val=&c4stat=ts_pct&c4comp=gt&c4val=&c5stat=&c5comp=gt&c5val=&order_by=date_game&order_by_asc=Y&offset="
    offsets = [i * 100 for i in xrange(25)]
    date = ''
    for offset in offsets:
        url = url_base + str(offset)
        r = requests.get(url)
        soup = BeautifulSoup(r.text, "html.parser")
        games = soup.findAll("tr", class_=[u''])[2:]
        break_from_outer_loop = False
        for game in games:
    #       For some reason <tr class=" thead"> satisfies class_=[u''] even though 
    #         "" != " thead" so I filter those out here. If you can figure out a fix
    #         so that those classes don't get picked up in games then please implement.
            if game["class"] == [u'', u'thead']:
                continue
            raw_game_stats = game.find_all("td")
            # get the date
            date = raw_game_stats[1].get_text()
            date = datetime.datetime(int(date[0:4]), int(date[5:7]), int(date[8:]))
            # don't collect this data if we haven't reached the start date and break inner loop
            if date < from_this_date:
                continue
            # don't collect this data if we've passed the until date and restart loop
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
    return data, target


def run_PCA(data, target):
    # do PCA stuff
    # This reduces the number of columns in the matrix but also changes the numbers.
    # I have no idea what the numbers mean or which columns were kept.
    pca = decomposition.PCA(n_components=5)
    pca.fit(data)
    princinpal_component_data = pca.transform(data)
    return princinpal_component_data, target
    pass


def simulation(season):
    # 2014-5 regular season started 10/28/2015 and ended 4/15/2015
    prev_start = datetime.datetime(season - 2, 10, 28)
    prev_end = datetime.datetime(season - 1, 04, 16)
    # Initial training set is all the games from 2014-5 season
    training_set, target = create_matrix(prev_start, prev_end, season-1)
    training_set, target = run_PCA(training_set, target)
    all_predictions = np.array([])
    model = naive_bayes.GaussianNB()
    model.fit(training_set, target)
    # 2015-6 regular season started 10/27/2015 and ended 4/13/2016
    current_date = datetime.datetime(season-1, 10, 27)
    next_date = datetime.datetime(season-1, 10, 28)
    end_date = datetime.datetime(season, 4, 14)
    while current_date <= end_date:
        daily_data, expected = np.array([]), np.array([])
        # Pull in games from just a single day
        # Loop because occasionally there are dates with no games
        while daily_data.size == 0:
            daily_data, expected = create_matrix(current_date, next_date, season)
            current_date = next_date
            next_date += datetime.timedelta(days=1)
        # Run PCA on daily data, make prediction, append it to our result set
        daily_data, expected = run_PCA(daily_data, expected)
        daily_predictions = model.predict(daily_data)
        np.append(all_predictions, daily_predictions)
        print(zip(daily_predictions, expected))
        # Replace n oldest entries in training/target sets with n games from today
        n, _ = daily_data.shape
        training_set = np.concatenate((training_set[n:], daily_data))
        target = np.concatenate((target[n:], expected))
        # Retrain the model with the actual outcomes of the day's games
        training_set, target = run_PCA(training_set, target)
        model.fit(training_set, target)
    return daily_predictions, target
    pass


def main():
    final_predictions, final_outcomes = simulation(2016)
    p, r, f1, _ = metrics.precision_recall_fscore_support(final_predictions,
                                                          final_outcomes,
                                                          average='binary')
    print(p, r, f1)
if __name__ == "__main__":
    main()
