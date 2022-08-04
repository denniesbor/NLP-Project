from random import random
from sklearn import tree
import streamlit as st
import warnings
import os
import subprocess
import datetime
import json
import re
from PIL import Image
from urllib.parse import quote
import gc
from dotenv import load_dotenv
from urllib.parse import quote

import pandas as pd
import numpy as np
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

# Viz Pkgs
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')

#Hide Warnings
warnings.filterwarnings("ignore")
st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout="wide")

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
db_user = os.getenv("user")
db_password = os.getenv("password")
db_host = os.getenv("host")
PORT = 3306
DATABASE = "nlp"
TABLENAME = "us_house_reps"

# name of the current
date_now = datetime.date.today()
file = 'data_' + str(date_now)

# navbar button editing
m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: rgb(18, 12, 125);
    color:white;
    font-size: 20px;
    border-radius: 10px;
}
</style>""", unsafe_allow_html=True)

# plotly object arguments
plotly_args = dict(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.8),
            margin={"r": 0, "t": 0, "l": 0, "b": 0})

# read from secret files
if (db_user==None):
    
    db_user = st.secrets["db_creds"]["db_user"]
    db_password = st.secrets["db_creds"]["db_password"]
    db_host = st.secrets["db_creds"]["db_host"]
    

def connect_db():
    ''' Function that creates a connection between the 
    Streamlit application and the AWS-hosted MySQL database
    
    -----------
    returns a queryset object
    '''
    try:
        print('Connecting database')
        engine = db.create_engine(f"mysql+pymysql://{db_user}:%s@{db_host}:{PORT}/{DATABASE}" % quote(db_password))
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
    df['Party'] = df['Party'].replace(['R','D',None], ['Republicans', 'Democrats','Other'], regex = True)
    df['Compound Sentiment'] = df['Compound Sentiment'].apply(float) 
    return df

# Extract topics from the topics dataframe
def topics():
    ''' Read the topics csv and return as a pandas dataframe, 
    and list of unique topics
    '''
    topic_df = pd.read_csv('topics.csv')
    topic_df['Topic'] = topic_df.Topic.str.title()
    topics = tuple(topic_df.Topic.unique())
    
    return topic_df, topics

# persist df in session_state
if 'df' not in st.session_state:
    try:
        df = pd.read_csv(f'{file}.csv')
    except FileNotFoundError:
        subprocess.run('rm data_20*',shell=True)
        df = connect_db()

        # drop duplicates
        df = df.drop_duplicates(subset = ['text'])
        df.reset_index(inplace=True, drop=False)
        # df.to_csv(f'{file}.csv', index=False)
        
    st.session_state['df'] = df

# topics
if 'topic_df' not in st.session_state:
    topic_df, topics = topics()
    st.session_state['topic_df_s'] = topic_df
    st.session_state['topics_s'] = topics

# stopwords
if 'stopwords' not in st.session_state:
    stopwords_set = set(STOPWORDS)

    # update stopwords set
    stopwords_set.update(['s','will','amp','must','rt'])
    st.session_state['stopwords'] = stopwords_set

# Query the entire df
def get_topic_df(topic:str, category:str=None, sentiment_type:list=[]) -> pd.DataFrame:
    ''' This function filters the tweets related to a specific topic, and sentiment value.
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
    df = st.session_state['df']
    
    # query topic associated n-grams
    topic = topic.title()
    topic_ = st.session_state['topic_df_s'].loc[st.session_state['topic_df_s']['Topic'] == topic]
    list_ngrams = topic_['n grams'].to_list()

    # query the daPartytaframe to extract tweets associated with topic ngrams
    topic_result = df[df.text.str.contains('|'.join(list_ngrams))]
    if(category):
        topic_result = topic_result[topic_result['Party'] == category]

    if (sentiment_type==[] or sentiment_type==['All'] or 'All' in sentiment_type):
        return topic_result
    else:
        sentiment_type = [sentiment.lower() for sentiment in sentiment_type]
        topic_result = topic_result[topic_result['Sentiment Value'].isin(sentiment_type)]
        
        return topic_result

def chart_type_plot(chart_type, df):
    """This function plots a specified chart of Histogram, bar chart,
    or Scatter plot and returns a plotly figure object

    params
    -------------
    input: str, pandas.Df
    chart type, pandas pandas dataframe
    returns
    ---------
    output -> plotly object
    a plotly figure object
    """

    args = dict(
        color="Party",
        template = 'plotly_dark',
        log_y = True
        )

    if chart_type == 'Histogram':
        fig = px.histogram(df,
                x="Compound Sentiment",
                histnorm = 'density',
                nbins = 20,
                text_auto = '.1f',
                **args)

    elif chart_type == "Bar":
        fig = px.histogram(df,
                x="Sentiment Value",
                barmode = 'group',
                text_auto = '.1f',
                **args)

    elif chart_type == "Scatter":
        df['Compound Sentiment'] = df['Compound Sentiment'].round(decimals=3)
        df = df[['Compound Sentiment','Party','tweet_id']].groupby(['Compound Sentiment','Party']).count()
        df = df.reset_index()
        fig = px.scatter(df, x='Compound Sentiment',
                            y='tweet_id',
                            labels={
                            "Compound Sentiment": "Sentiment Label (-1 - 1)",
                            "tweet_id": "Value Count"
                            },
                            **args)

    fig.update_layout(**plotly_args)

    return fig

# plot tweet sentiment timeseries
def get_time_sentiments(df):

  '''This function calculates the volume of neutral, negative, 
  and positive tweets since the start of the year 2022
  
  parameters
  -------------
  input: pandas.DataFrame
    pandas dataframe object

  return: object
    plotly object
  
  '''
  
  df["created_at"] = pd.to_datetime(df.created_at)
  df = df[df["created_at"] >= '2022-01-01']
  df = df.resample('D', on='created_at')["Sentiment Value"].value_counts().unstack(1)
  df.reset_index(inplace=True)
  df = df.melt("created_at", var_name='sentiment',  value_name='vals')
  fig = px.line(df, x="created_at", y="vals", color='sentiment',
                labels={
                    'created_at':'Date',
                    'vals':'Count'
                })
  fig.update_layout(**plotly_args)

  return fig

# plot word cloud
def plot_wc(df, mask=None, max_words=200, max_font_size=100, figure_size=(8,12), color = 'white',
                   title = None, title_size=40, image_color=False):
  """This function extracts tweets from a dataframe, split into individual terms
  and plots the most occuring words. Stopwords are excluded
  
  params
  --------
  input: dict:pd.DataFrame
    pandas dataframe
  returns
  ---------
  output -> plt.Figure
    matplotlib figure object
  """

  fig = plt.figure(figsize=figure_size,dpi=300)
  text = " ".join(tweet for tweet in df["text"])
  wordcloud = WordCloud(background_color=color,
                  stopwords = st.session_state['stopwords'],
                  max_words = max_words,
                  max_font_size = max_font_size, 
                  random_state = 42,
                  width=400, 
                  height=500,
                  mask = mask)
  wordcloud.generate(str(text))
  if image_color:
    image_colors = ImageColorGenerator(mask);
    plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");
    plt.title(title, fontdict={'size': title_size,  
                              'verticalalignment': 'bottom'})
  else:
    plt.imshow(wordcloud,interpolation="bilinear");
  plt.axis("off")
  
  return fig

# tree map
def tree_map(df):
    ''' function which takes a dataframe of tweets and returns tree map
        as a Plotly object

    params:
    -----------
    input: pandas dataframe
        df

    returns
    ----------
    output: obj
        plotly figure
    '''

    all_words = [item for sublist in [word.split(' ') for word in list(df.text)] for item in sublist if item not in st.session_state['stopwords']]
    all_words=pd.Series(np.array(all_words))
    common_words=all_words.value_counts()[:70].rename_axis('Common Words').reset_index(name='count')
    fig = px.treemap(common_words, path=['Common Words'], values='count',width=800, height=400)

    fig.update_layout(**plotly_args)
    
    return fig

def main():

    html_temp = """
	<div style="background-color:#1a2f52;"><p style="color:white;font-size:30px;padding:9px">Visualization of Tweets from US House Representatives</p></div>
	"""
    st.markdown(html_temp, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,1,1])

    with col1:
        cat_all = st.button('ALL')
    with col2:
        cat_dem = st.button('Democrats')
    with col3:
        cat_rep = st.button('Republicans')
    
    categories = [None, 'Democrats','Republicans']
    
    cat = [(i, cat) for i,cat in enumerate([cat_all,cat_dem, cat_rep]) if cat != False]
    if(cat):
        cat = categories[cat[0][0]]
    else: cat = None
    
    # type
    if(cat==None):
        type = 'All'
    else:
        type= cat
    with st.spinner(f"Please wait..."):
        col4, col5, col6 = st.columns([1,3,3])
        with col4:
            chart_type = st.selectbox("Chart Type", ('Histogram','Bar','Scatter'), 1)
            option = st.selectbox("Area", st.session_state['topics_s'], 1)
            sentiment_type = st.multiselect("Sentiment Type", ('All','Positive','Negative', 'Neutral'))
            top_df = get_topic_df(option, cat,sentiment_type)
            st.metric(label="Number of Tweets", value=len(top_df))
            
        with col5:
            st.markdown(f"<h3 style='text-align: center; color: white; font-size:20px; background-color:black;'>Sentiment Distribution({option}-{type})</h3>", unsafe_allow_html=True)
            fig = chart_type_plot(chart_type,top_df)  
            st.plotly_chart(fig, use_container_width=True,config = {'displayModeBar': False})
        
        with col6:
            st.markdown(f"<h3 style='text-align: center; color: white; background-color:black; font-size:20px;'>Time Based Sentiment Sentiments({option}-{type})</h3>", unsafe_allow_html=True)
            fig = get_time_sentiments(top_df)
            # st.pyplot(fig_wc)
            st.plotly_chart(fig, use_container_width=True,config = {'displayModeBar': False})

        col7, col8, col9 = st.columns([3,2,3])
        
        with col7:
            tweets = top_df.sample(n=100, random_state=101)
            st.markdown(f"<h3 style='text-align: center; color: white; background-color:black; font-size:20px;'>Sample {len(tweets)} Tweets({option}-{type})</h3>", unsafe_allow_html=True)
            tweets.reset_index(drop=True, inplace=True)
            st.dataframe(tweets[['Party','text']])
            
        with col8:
            st.markdown(f"<h3 style='text-align: center; color: white; background-color:black; font-size:20px;'>Word Cloud({option}-{type})</h3>", unsafe_allow_html=True)
            pos_mask = np.array(Image.open('us_1.PNG'))
            fig_wc = plot_wc(top_df,mask=pos_mask,color='white',max_font_size=100,title_size=30)
            st.pyplot(fig_wc)
            
        with col9:
            st.markdown(f"<h3 style='text-align: center; color: white; background-color:black; font-size:20px;'>Tree Map of 70 Common Words({option}-{type})</h3>", unsafe_allow_html=True)
            fig = tree_map(top_df)
            st.plotly_chart(fig, use_container_width=True,config = {'displayModeBar': False})
   
    if st.button("Exit"):
        st.balloons()


if __name__ == '__main__':
    main()
