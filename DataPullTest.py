'''
Created on Mar 16, 2016
@author: Alex
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
    return data, target


def run_PCA(data):
    # do PCA stuff
    # This reduces the number of columns in the matrix but also changes the numbers.
    # I have no idea what the numbers mean or which columns were kept.
    game_id = np.array([[float(row[0]), float(row[1])] for row in data])
    pca = decomposition.PCA(n_components=10)
    pca.fit(data)
    #print("shape pre transform is {}").format(data.shape)
    princinpal_component_data = pca.transform(data)
    #print("shape after transform is {}").format(princinpal_component_data.shape)
    
    for i, row in enumerate(princinpal_component_data):
        row[0] = game_id[i][0]
        row[1] = game_id[i][1]
    return princinpal_component_data
    pass


def group(matrix):
    # Group game stats by team_id for the purpose of averaging
    team_histories = {float(i): None for i in xrange(30)}
    for row in matrix:
        team_id = float(row[0])
        game_stats = row[2:30]
        if team_histories[team_id] is None:
            team_histories[team_id] = game_stats
        else:
            team_histories[team_id] = np.concatenate((team_histories[team_id], game_stats))
    # Concatenate is flattening the array unexpectedly
    # Have to reshape each matrix to get the data in correct form.
    # If someone sees a fix, please add it
    for k, v in team_histories.items():
        team_histories[k] = np.reshape(v, (-1, 28))
    return team_histories
    pass


def construct_validation_data(daily_data, team_histories):
    #Given daily games data, return averages of teams' previous games
    validation_data = np.array([])
    for game in daily_data:
        team1_id, team2_id = game[0], game[1]
        team1_avg = np.mean(team_histories[team1_id], axis=0)
        team2_avg = np.mean(team_histories[team2_id], axis=0)[1:]
        all_avgs = np.append([team1_id, team2_id], [team1_avg])
        all_avgs = np.append(all_avgs, team2_avg)
        validation_data = np.concatenate((validation_data, all_avgs))
        
    validation_data = np.reshape(validation_data, (-1, 57))

    return validation_data
    pass


def update_team_histories(daily_data, team_histories):
    for game in daily_data:
        team_id = float(game[0])
        game_stats = game[2:30]
        team_histories[team_id] = np.append(team_histories[team_id], game_stats)
    for k, v in team_histories.items():
        team_histories[k] = np.reshape(v, (-1, 28))
    return team_histories
    pass

def simulation(season):
    # 2014-5 regular season started 10/28/2015 and ended 4/15/2015 (Consider not every season starts/ends on the same day)
    prev_start = datetime.datetime(season - 2, 10, 28)
    prev_end = datetime.datetime(season - 1, 04, 16)
    # Initial training set is all the games from 2014-5 season
    training_set, target = create_matrix(prev_start, prev_end, season-1)
    # Group games by team for purposes of averaging
    team_histories = group(training_set)
    pca = decomposition.PCA(n_components=20)
    training_set = pca.fit_transform(training_set) #'added PCA'
    print("Training set is size {}").format(training_set.shape)
    all_predictions = np.array([])
    all_targets = np.array([])
    model = naive_bayes.GaussianNB()
    model.fit(training_set, target)
    # 2015-6 regular season started 10/27/2015 and ended 4/13/2016
    current_date = datetime.datetime(season-1, 10, 27)
    next_date = datetime.datetime(season-1, 10, 28) #changed season dates to decrease runtime for testing, change back
    end_date = datetime.datetime(season-1, 12, 14)
    while current_date <= end_date:
        daily_data, expected = np.array([]), np.array([])
        # Pull in games from just a single day
        # Loop because occasionally there are dates with no games
        while daily_data.size == 0:
            daily_data, expected = create_matrix(current_date, next_date, season)
            current_date = next_date
            next_date += datetime.timedelta(days=1)
            print("shape of daily data is {}").format(daily_data.shape)
        # Construct the averages to be used as validation data
        validation_data = construct_validation_data(daily_data, team_histories)
        # Run PCA on validation data
        
        validation_data = pca.transform(validation_data) #'running PCA'
        print('shape of PCAed validation data is {}').format(validation_data.shape)
        
        daily_predictions = model.predict(validation_data)
        print("daily predictions are {}").format(daily_predictions)
        all_predictions = np.append(all_predictions, daily_predictions)
        all_targets = np.append(all_targets, expected)
        print(current_date)
        print(zip(daily_predictions, expected))
        # Replace n oldest entries in training/target sets with n games from today
        PCA_daily_data = pca.transform(daily_data)
        n, _ = daily_data.shape
        training_set = np.concatenate((training_set[n:], PCA_daily_data))
        target = np.concatenate((target[n:], expected))
        # Add the day's games to team histories for future averages
        team_histories = update_team_histories(daily_data, team_histories)
        # Retrain the model with the actual outcomes of the day's games
        # Don't think we need rerun PCA on rows that have already been PCAed
        "training_set = run_PCA(training_set) #running pca"
        model.fit(training_set, target)
    return all_predictions, all_targets
    pass


def main():
    
    final_predictions, final_targets = simulation(2016)
    
    p, r, f1, _ = metrics.precision_recall_fscore_support(final_predictions,
                                                          final_targets,
                                                          average='binary')
    print(p, r, f1)

if __name__ == "__main__":
    main()
