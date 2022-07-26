import re
import pandas as pd

url = 'https://pressgallery.house.gov/member-data/members-official-twitter-handles'

def get_house_reps():

  '''A function which scrapes US representatives and their Twitter handles
  
  ------------
  attributes

  return: list
    list of tuples of each house rep and the party
  ------------
  '''

  # read the housereps and pass into a dataframe
  print('***Fetching house reps ***')
  dfs = pd.read_html(url)
  print('***House reps response received***')
  house_reps = dfs[0]

  # make the first row as columns
  house_reps.columns = house_reps.iloc[0]

  df = house_reps.drop(index=0, inplace=False)[['Twitter Handle','Party']]
  df['Twitter Handle'] = df['Twitter Handle'].str.replace('@', '')

  # create list of tuples from the columns of dataframes
  house_rep_lists = list(zip(df['Twitter Handle'], df.Party))
  
  return house_rep_lists