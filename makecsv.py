import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
#from pandas.tools.plotting import scatter_matrix
#'data/churn_train.csv'
def clean_data(df):
    print(df.info())
    print(df.describe().T)

    #make our y dataframe
    y = pd.to_datetime(df['last_trip_date'])
    #convert datetime, and measure days since last Use
    #determine if we've churned
    churntime = (datetime(2014, 7, 1) - (y))
    targets = np.zeros(len(churntime))
    #datetime format has .days, so we have to convert to a number...had to do it in a loop
    for i in range(0,len(churntime)):
        targets[i] = churntime[i].days > 30
    #take out last trip date because it's embedded in the target
    df.pop('last_trip_date')
    df.pop('signup_date') #data set limited to sign ups in January

    #create categorical on whether rating of driver is missing
    df['unrated'] = df['avg_rating_of_driver'].isnull()
    #fill in NaNs with mean values
    df['avg_rating_by_driver'].fillna(np.mean(df['avg_rating_by_driver']), inplace = True)
    df['avg_rating_of_driver'].fillna(np.mean(df['avg_rating_of_driver']), inplace = True)
    #rename Nan from missing phone info
    df['phone'].fillna('unknown', inplace = True)
    #create total distance column
    df['total_distance'] = df['avg_dist'] * df['trips_in_first_30_days']

    #make categoricals on cities and phone types
    categoricals = pd.get_dummies(df[['city','phone']])
    df.pop('city')
    df.pop('phone')

    #remerged our dataframes into one
    df = pd.merge(df, categoricals, left_index=True, right_index=True)
    df_features = df  #another name for clarity
    df_targets = pd.DataFrame(targets)
    #return features and target dataframes
    return df_features, df_targets


'''
Investigation into important features
'''

# print(df.info())
# print(df.describe().T)
# #print(churntime, targets)
# # churn = churntime.values > 30
# # print(churn)
# # print(y[0])
#
#
#
#
# scatterall =  ['avg_dist', 'avg_rating_by_driver', 'avg_rating_of_driver', 'avg_surge', 'surge_pct', 'trips_in_first_30_days', 'luxury_car_user', 'weekday_pct','city_Astapor',"city_King's Landing", 'city_Winterfell', 'phone_Android', 'phone_iPhone', 'phone_unknown', 0]
#
# scatter1 =  ['avg_dist', 'avg_rating_by_driver', 'avg_rating_of_driver', 'avg_surge', 'surge_pct', 'trips_in_first_30_days', 'weekday_pct', 0]
#
# df_investigate = pd.merge(df, df_investigate, left_index=True, right_index=True)
# scatter_matrix1 = pd.scatter_matrix(df_investigate[scatter1].sample(n=10000))
#
# for ax in scatter_matrix1.ravel():
#     ax.set_xlabel(ax.get_xlabel(), fontsize = 8, rotation = 90)
#     ax.set_ylabel(ax.get_ylabel(), fontsize = 8, rotation = 0)
# plt.savefig('scatter1.jpg')
#
# mycorrs = df_investigate.corr()[0]
# #mycorrsorder = mycorrs.order(kind="quicksort")
# print(mycorrs)
# print(df_investigate.head())


if __name__ == '__main__':
    df_train = pd.read_csv('data/churn_train.csv')
    features_train, targets_train = clean_data(df_train)
    df_test = pd.read_csv('data/churn_test.csv')
    features_test, targets_test = clean_data(df_test)
    dfout = pd.merge(features_test, targets_test, left_index=True, right_index=True)
    dfout.to_csv('dfout.csv')

    mycorrs = dfout.corr()[0]
    #mycorrsorder = mycorrs.order(kind="quicksort")
    print(mycorrs)
    #print(df_investigate.head())
