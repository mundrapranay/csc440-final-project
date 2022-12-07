import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


TEAMS = ['Chelsea FC', 'Bayern Munich', 'FC Barcelona', 'LOSC Lille', 'Juventus FC']


def load_data():
    net_spending = []
    for team in TEAMS:
        data = pd.read_csv('./data/transfer_2010-2020.csv')
        season_start_year = 2020
        team_name = team
        season_mask = data['season_start_year'] == season_start_year
        bought_mask = data['squad'] == team_name
        sold_mask = data['joined_from'] == team_name
        transfer_data = data[season_mask]
        sold_data = transfer_data[sold_mask]
        bought_data = transfer_data[bought_mask]
        sold_data = sold_data.drop(['comp_name','region','country','season_start_year','current_club','player_num','player_dob','player_age','player_nationality','player_height_mtrs','player_foot','player_url'], axis=1)
        bought_data = bought_data.drop(['comp_name','region','country','season_start_year','current_club','player_num','player_dob','player_age','player_nationality','player_height_mtrs','player_foot','player_url'], axis=1)
        sold_data.to_csv('./data/2020_sold_{0}.csv'.format(team_name))
        bought_data.to_csv('./data/2020_bought_{0}.csv'.format(team_name))
        spent_money = bought_data['player_market_value_euro'].sum()
        earned_money = sold_data['player_market_value_euro'].sum()
        # print('Spent {0} : {1}'.format(team_name, spent_money))
        # print('Earned {0} : {1}'.format(team_name, earned_money))
        # print('Net Profit/Loss : {0}'.format(earned_money - spent_money))
        net_spending.append(earned_money - spent_money)
    # plt.bar(TEAMS, net_spending)
    # plt.xlabel('Team Name')
    # plt.ylabel('Net Profit/Loss')
    # plt.tight_layout()
    # plt.show()
    # plt.savefig('./figures/net_spending.png')
    # plt.cla()
    for t, m in zip(TEAMS, net_spending):
        print('{0} : {1:.1E}'.format(t, m))




if __name__ == '__main__':
    load_data()