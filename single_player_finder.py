import pandas as pd 
import numpy as np 
import argparse
import os 
import faiss


DROP_COLS = []
DATA_LOC = './data/{0}_data_fb_ref.csv'


'''
1. Load the data
2. normalize the numeric columns using : z-score normalization
3. build the data matrix and idMap
'''
def load_and_preprocess_data(params):
    data = pd.read_csv(DATA_LOC.format(params.league))
    data_season_mask = data['Season'] == params.season
    data = data[data_season_mask]
    data_numeric_cols = data.select_dtypes(include='number')
    data_numeric_cols_normalized = (data_numeric_cols - data_numeric_cols.mean()) / (data_numeric_cols.std())
    data[data_numeric_cols.columns] = data_numeric_cols_normalized
    data =  data.fillna(0.0)
    data['Player'] = data['Player Lower']
    data['Pos'] = data['Position Grouped']
    # print(data.head())
    data = data.drop(['Season','Matches','Squad','Nation', 'Comp', 'Age', 'Born', 'League ID', 'League Name', 'Team Name', 'Team Country', 'First Name Lower', 'Last Name Lower','First Initial Lower','Team Country Lower','Nationality Code','Nationality Cleaned', 'Outfielder Goalkeeper', 'Player Lower', 'Position Grouped'], axis=1)
    # print(data.head())
    # print(data.keys()[28])
    # for row in data.iterrows():
    dataList = data.values.tolist()
    # 1 : player name, 2 : position, [3:] : values for attributes : vector     
    data_matrix = np.zeros((len(dataList), 182))
    idMap = [None] * len(dataList)
    labelMap = [None] * len(dataList)
    for idx, l in enumerate(dataList):
        player_name = l[1]
        player_vector = l[3:]
        idMap[idx] = player_name
        data_matrix[idx] = player_vector

    data_matrix = data_matrix.astype('float32')
    return data_matrix, idMap


def build_index(data):
    N = len(data)
    d = 182
    data[:, 0] += np.arange(N) / 1000.
    index = faiss.IndexFlatL2(d)
    print(index.is_trained)
    index.add(data)
    print(index.ntotal)
    return index 

def player_finder(index, idMap, data, params):
    params.league = 'premier_league'
    query_data, query_idMap = load_and_preprocess_data(params)
    player_name = 'kevin de bruyne'
    idx = query_idMap.index(player_name)
    print(idx)
    # D, I = index.search(query_data[:5], params.k)
    query_kdb = np.zeros((1, 182))
    query_kdb[0] = query_data[idx]
    query_kdb = query_kdb.astype('float32')
    D, I = index.search(query_kdb, params.k)
    print(I)
    first_search = query_idMap[:5]
    for idx, nbs in enumerate(I):
        similar_players = [idMap[n] for n in nbs]
        print('Query player : {0}\t Similar Players : {1}'.format(player_name, ','.join(similar_players)))
    # print(D)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--league', type=str, default='premier_league', help='name of league to build index over = [premier_league, la_liga, bundesliga, ligue_1, serie_a]')
    parser.add_argument('--season', type=str, default='2018-2019', help='season for which we are searing = [2017-2018, 2018-2019, 2019-2020, 2020-2021, 2021-2022]')
    parser.add_argument('--k', type=int, default=10, help='k for the top-k search')
    params = parser.parse_args()

    data, idMap = load_and_preprocess_data(params)
    index = build_index(data)
    player_finder(index, idMap, data, params)