
#Vader Sentiment Analysis

import nltk
import re

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA

sid = SIA()


def compute_sentiments(df):

  """Function which computes the sentiments of a dataframe texts. 
  """
  df['sentiments'] = df['text'].apply(lambda x: sid.polarity_scores(' '.join(re.findall(r'\w+',x.lower()))))

  # extract scores of sentiments. 0.00001 added incase of a score of 0
  df['Positive Sentiment'] = df['sentiments'].apply(lambda x: x['pos']+1*(10**-6)) 
  df['Neutral Sentiment'] = df['sentiments'].apply(lambda x: x['neu']+1*(10**-6))
  df['Negative Sentiment'] = df['sentiments'].apply(lambda x: x['neg']+1*(10**-6))
  df['Compound Sentiment'] = df['sentiments'].apply(lambda x: x['compound']+1*(10**-6))
  df['Sentiment Value'] = df['Compound Sentiment'].apply(lambda x: 'positive' if x > 0.05 else ('negative' if x < -0.05 else 'neutral'))
  df.drop(columns=['sentiments'],inplace=True)

  print('Finished computing sentiment analysis')
  
  return df


def clean_tweets(tweet:str):

  """ This function cleans the tweets

  Attrs
  ---------
  input: str
    tweet
  Returns
  ---------
  output: str
    clean tweet 
  """
  
  tweet = tweet.lower()

  #Remove twitter handlers
  tweet = re.sub('@[^\s]+','',tweet)

  #remove hashtags
  tweet = re.sub(r'\B#\S+','',tweet)

  # Remove URLS
  tweet = re.sub(r"http\S+", "", tweet)

  #remove all single characters
  tweet = re.sub(r'\s+[a-zA-Z]\s+', '', tweet)

  # Substituting multiple spaces with single space
  tweet = re.sub(r'\s+', ' ', tweet, flags=re.I)

  # Remove all the special characters
  tweet = ' '.join(re.findall(r'\w+', tweet))

  return tweet


