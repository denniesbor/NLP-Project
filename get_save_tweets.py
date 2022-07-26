import os 
import warnings
from urllib.parse import quote

import pandas as pd
import numpy as np

import tweepy
from tweepy.errors import TweepyException

import pymysql
from sqlalchemy import create_engine
from sqlalchemy.exc import ProgrammingError, OperationalError
from sqlalchemy import create_engine, MetaData, DateTime,\
    DECIMAL, Text,\
        Table, Column,\
            Integer, String
            
from sqlalchemy import MetaData

from dotenv import load_dotenv

# custom modules
from sentiment_analysis import compute_sentiments, clean_tweets
from twitter_api import api
from fetch_house_reps import get_house_reps

pymysql.install_as_MySQLdb()


#load secret files
load_dotenv()

# load db params
meta = MetaData()

USER = os.getenv("user")
PASSWORD = os.getenv("password")
HOST = os.getenv("host")
PORT = 3306
DATABASE = "nlp"
TABLENAME = "us_house_reps"

railaruto = Table(
    f'{TABLENAME}',meta,
    Column('id', Integer, primary_key = True, autoincrement = True),
    Column('tweet_id', String(255)),
    Column('username', String(255)),
    Column('Party', String(50)),
    Column('text', Text(500)), 
    Column('favorite_count', Integer), 
    Column('retweet_count', Integer),
    Column('created_at', DateTime), 
    Column('source', String(255)), 
    Column('retweets', Integer),
    Column('favorites', Integer), 
    Column('Positive Sentiment', DECIMAL(15,4)),
    Column('Neutral Sentiment', DECIMAL(15,4)),
    Column('Negative Sentiment', DECIMAL(15,4)), 
    Column('Compound Sentiment', DECIMAL(15,4)),
    Column('Sentiment Value', String(255))
)

# sql database setup
engine = create_engine(f"mysql://{USER}:%s@{HOST}:{PORT}" % quote(PASSWORD))
connection = engine.connect()


def dataframe_tosql(df):
    ''' Function that establishes connection to a mysql database
    and save the data frame
    '''
    
    print('beginning tweets export to a sql server')
    
    try:
        engine.execute(f"CREATE DATABASE {DATABASE}")
    except ProgrammingError:
        warnings.warn(
            f"Could not create database {DATABASE}. Database {DATABASE} may already exist."
        )
        pass

    engine.execute(f"USE {DATABASE}")
    
    # engine.execute(f"DROP TABLE IF EXISTS {TABLENAME}")
    try:
        # create a new database
        meta.create_all(engine)
        print(f'created table {TABLENAME}')
    except OperationalError:
        warnings.warn(
            f"Table {TABLENAME} exists. Skipping!!!!"
        )
        pass
   
    df.to_sql(name=TABLENAME, schema=DATABASE, if_exists='append', con=engine, index = False)

    print('Tweets exported to a sql server')

def get_tweets(username, party, number_of_tweets= 1000):
    
    ''' A function that fetch 3200 tweets from a user and return as pandas DF
    
    '''

    tweet_list = []

    print(f"Fetching tweets of {username}")
    
    #get tweets
    try:
        for tweet in tweepy.Cursor(api.user_timeline, screen_name = username).items(number_of_tweets):
            
            tweet_id = tweet.id_str # unique integer identifier for tweet
            favorite_count = tweet.favorite_count
            retweet_count = tweet.retweet_count
            created_at = tweet.created_at # utc time tweet created
            source = tweet.source # utility used to post tweet
            # reply_to_status = tweet.in_reply_to_status_id # if reply int of orginal tweet id
            # reply_to_user = tweet.in_reply_to_screen_name # if reply original tweets screenname
            retweets = tweet.retweet_count # number of times this tweet retweeted
            favorites = tweet.favorite_count # number of time this tweet liked

            # clean the tweet
            text = clean_tweets(tweet.text) # utf-8 text of tweet
            # append attributes to list

            tweet_list.append({'tweet_id':tweet_id,
                                'username':username,
                                'Party': party,
                                'text':text, 
                                'favorite_count':favorite_count,
                                'retweet_count':retweet_count,
                                'created_at':created_at, 
                                'source':source, 
                                # 'reply_to_status':reply_to_status, 
                                # 'reply_to_user':reply_to_user,
                                'retweets':retweets,
                                'favorites':favorites})
    except TweepyException:
        
        print('Connection errors')
        
        get_tweets(username, party,number_of_tweets)
        
        
    
    print(f"Finished fetching tweets of {username}")
    print('/n')
    print('/n')
    print(f"EXporting {username} tweets to a pandas dataframe")
    
    # create dataframe
    df = pd.DataFrame(tweet_list)
    # compute sentiments of tweets
    df = compute_sentiments(df)
    dataframe_tosql(df)
    
    return

if __name__ == '__main__':
    
    number_of_tweets = 3
    
    # users to fetch tweets from
    # test = [('williamsruto','uda'),('railaodinga','azimio')]
    house_reps = get_house_reps()
    
    for user, party in house_reps:
        get_tweets(user, party, number_of_tweets)
    #set count to however many tweets you want
    
    # for user,party in test:
    #     get_tweets(user, party, number_of_tweets)
    
    # remove duplicates
    engine.execute(
        f'''
        DELETE c1 FROM {TABLENAME} c1
        INNER JOIN {TABLENAME} c2 
        WHERE
        c1.id > c2.id AND 
        c1.tweet_id = c2.tweet_id'''     
        )
    
    