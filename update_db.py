import os
import datetime
import pandas as pd
import numpy as np

import tweepy
from tweepy.errors import TweepyException

# custom modules
from sentiment_analysis import compute_sentiments, clean_tweets
from twitter_api import api
from fetch_house_reps import get_house_reps
from get_save_tweets import dataframe_tosql


#acquire time tweets
date_time = datetime.datetime.now()
time_delta1 = datetime.timedelta(hours=4)
date_since = date_time-time_delta1

# extract unix time
unix_time_since = datetime.datetime.timestamp(date_since)
print(unix_time_since)

def get_tweets(username, party, number_of_tweets= 1000):
    
    ''' A function that fetch 3200 tweets from a user and return as pandas DF
    
    '''

    tweet_list = []

    print(f"Fetching tweets of {username}")
    
    #get tweets
    try:
        for tweet in tweepy.Cursor(api.user_timeline, screen_name = username, tweet_mode='extended').items(number_of_tweets):
            
            created_at = tweet.created_at # utc time tweet created
            unix_time_created = datetime.datetime.timestamp(created_at)
            
            if unix_time_created >= unix_time_since:
            
                tweet_id = tweet.id_str # unique integer identifier for tweet
                favorite_count = tweet.favorite_count
                retweet_count = tweet.retweet_count
                source = tweet.source # utility used to post tweet
                retweets = tweet.retweet_count # number of times this tweet retweeted
                favorites = tweet.favorite_count # number of time this tweet liked

                # clean the tweet
                text = clean_tweets(tweet.full_text) # utf-8 text of tweet
                # append attributes to list

                tweet_list.append({'tweet_id':tweet_id,
                                    'username':username,
                                    'Party': party,
                                    'text':text, 
                                    'favorite_count':favorite_count,
                                    'retweet_count':retweet_count,
                                    'created_at':created_at, 
                                    'source':source, 
                                    'retweets':retweets,
                                    'favorites':favorites})
            else:
                break
            
    except TweepyException:
        
        print('Connection errors')
        
        get_tweets(username, party,number_of_tweets)
    
    else:
        pass
        

    print(f"Finished fetching tweets of {username}")
    print('\n')
    print('\n')
    print(f"EXporting {username} tweets to a pandas dataframe")
    
    if tweet_list == []:
        pass
    else:
        # create dataframe
        df = pd.DataFrame(tweet_list)
        # compute sentiments of tweets
        df = compute_sentiments(df)
        dataframe_tosql(df)
    
if __name__ == '__main__':
    house_reps = get_house_reps()
    
    for user, party in house_reps:
        if user != 'teammoulton':
            get_tweets(user, party)
        else:
            pass
