import os
import tweepy
from dotenv import load_dotenv

load_dotenv()

# load secret files
api_key = os.getenv('api_key')
api_key_secret = os.getenv('api_key_secret')
access_token = os.getenv('access_token')
access_token_secret = os.getenv('access_token_secret')
bearer_token = os.getenv('bearer_token')

# Authenticate 
auth = tweepy.OAuth1UserHandler(api_key,api_key_secret)
auth.set_access_token(access_token,access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)