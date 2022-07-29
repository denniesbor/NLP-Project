import os
import gc
import sqlalchemy as db
from dotenv import load_dotenv
from urllib.parse import quote
import pandas as pd

# import database database
import sqlalchemy as db
from sqlalchemy import Table, Column, INTEGER, MetaData
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

import plotly
import plotly.express as px
import plotly.graph_objs as go
import plotly.offline as pyo
from plotly.subplots import make_subplots

#load secret files
load_dotenv()

# configure db info
USER = os.getenv("user")
PASSWORD = os.getenv("password")
HOST = os.getenv("host")
PORT = 3306
DATABASE = "nlp"
TABLENAME = "us_house_reps"

def connect_db():
    ''' Function that creates a connection between the 
    Streamlit application and the AWS-hosted MySQL database
    
    -----------
    returns a queryset object
    '''
    try:
        print('Connecting database')
        engine = db.create_engine(f"mysql+pymysql://{USER}:%s@{HOST}:{PORT}/{DATABASE}" % quote(PASSWORD))
        connection = engine.connect()
        metadata = db.MetaData()
        housereps = db.Table(f'{TABLENAME}', metadata, autoload=True, autoload_with=engine)

        # CREATE THE SESSION OBJECT
        Session = sessionmaker(bind=engine)
        session = Session()

        # make queries
        query = session.query(housereps).with_entities(housereps.c.tweet_id, \
                                                    housereps.c.Party, housereps.c.text, \
                                                    housereps.c.created_at, housereps.c['Compound Sentiment'],\
                                                    housereps.c['Sentiment Value'])
        print('DB connection success')
    except:
        print('DB connection error')
        connect_db()
    else:
      pass
    
    # convert query result to a dataframe
    print('Exporting queryset to a pd dataframe')
    df = pd.DataFrame(query)
    df.columns = query[0].keys()

    # remove empty tweets
    df = df[df.text != '']
    df['Party'] = df['Party'].replace(['R','D',None], ['Republicans', 'Democrats','Other'], inplace = True)
    df['Compound Sentiment'] = df['Compound Sentiment'].apply(float) 
    return df

try:
  df = pd.read_csv('all_tweets.csv')
except FileNotFoundError:
    df = connect_db()

    # drop duplicates
    df = df.drop_duplicates(subset = ['text'])
    df.reset_index(inplace=True, drop=False)
    df.to_csv('all_tweets.csv', index=False)

'''
Extract topics from the topics dataframe
'''

def topics():
    ''' Read the topics csv and return as a pandas dataframe, 
    and list of unique topics
    '''
    topic_df = pd.read_csv('topics.csv')
    topics = tuple(topic_df.Topic.unique())
    
    return topic_df, topics

# topics
topic_df, topics = topics()

'''
Query the entire df
'''

def get_topic_df(topic:str, category:str=None):
    ''' This function filters the tweets related to a specific topic.
    The topic is passed into topics df to extract ngrams associated with it. 
    The tweets are then extracted from the main dataframe using the ngrams.
    Category is none if all the tweets from the democrats
    and republicans are required.

    parameters
    ---------------
    input: topic: str, category: str default: None

    returns: pd.DataFrame
    filtered dataframe
    ----------------
    '''
    # query topic associated n-grams
    topic = topic.title()
    topic_ = topic_df.loc[topic_df['Topic'] == topic]
    list_ngrams = topic_['n grams'].to_list()
    print(list_ngrams)

    # query the daPartytaframe to extract tweets associated with topic ngrams
    topic_result = df[df.text.str.contains('|'.join(list_ngrams))]

    if(category):
        topic_result = topic_result[topic_result['Party'] == category]

    return topic_result


