import streamlit as st
import warnings
import os
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

print(os.environ)

if (db_user==None):
    
    db_user = os.environ["db_user"]
    db_password = os.environ["db_password"]
    db_host = os.environ["db_host"]
    
print(db_user,db_host)

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
        df = pd.read_csv('all_tweets.csv')
    except FileNotFoundError:
        df = connect_db()

        # drop duplicates
        df = df.drop_duplicates(subset = ['text'])
        df.reset_index(inplace=True, drop=False)
        df.to_csv('all_tweets.csv', index=False)
        
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
def get_topic_df(topic:str, category:str=None) -> pd.DataFrame:
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
    df = st.session_state['df']
    
    # query topic associated n-grams
    topic = topic.title()
    topic_ = st.session_state['topic_df_s'].loc[st.session_state['topic_df_s']['Topic'] == topic]
    list_ngrams = topic_['n grams'].to_list()

    # query the daPartytaframe to extract tweets associated with topic ngrams
    topic_result = df[df.text.str.contains('|'.join(list_ngrams))]

    if(category):
        topic_result = topic_result[topic_result['Party'] == category]

    return topic_result

def main():

    html_temp = """
	<div style="background-color:#1a2f52;"><p style="color:white;font-size:40px;padding:9px">Visualization of Tweets from US House Representatives</p></div>
	"""
    st.markdown(html_temp, unsafe_allow_html=True)

    option = st.sidebar.selectbox("Which Category?", st.session_state['topics_s'], 1)
    print(option)
    col1, col2, col3 = st.columns([1,1,1])

    with col1:
        cat_all = st.button('ALL')
    with col2:
        cat_dem = st.button('Democrats')
        if cat_dem:
            st.write("Hello Dems")
    with col3:
        cat_rep = st.button('Republicans')
    
    categories = [None, 'Democrats','Republicans']
    
    cat = [(i, cat) for i,cat in enumerate([cat_all,cat_dem, cat_rep]) if cat != False]
    if(cat):
        cat = categories[cat[0][0]]
    else: cat = None
    print(cat)
    
    top_df = get_topic_df(option, cat)
    # st.table(top_df.head(10))
    
    col1, col2, col3 = st.columns([2,6,4])
    with col1:
        chart_type = st.selectbox("chart_type", ('Histogram','Bar'), 1)
    with col2:
        st.markdown("<h3 style='text-align: center; color: white;'>Sentiment Distribution</h3>", unsafe_allow_html=True)
        if chart_type == 'Histogram':
            fig = px.histogram(top_df,
                   x="Compound Sentiment",
                   histnorm = 'percent',
                   color="Party",
                   template = 'plotly_dark',
                   text_auto = '.1f')
        else:
            fig = px.histogram(top_df,
                   x="Sentiment Value",
                   color="Party",
                   template = 'plotly_dark',
                   barmode = 'group',
                   text_auto = '.1f')
             
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        st.markdown("<h3 style='text-align: center; color: white;'>Word Cloud</h3>", unsafe_allow_html=True)
        # wordcloud
        text = " ".join(tweet for tweet in top_df["text"])
        
        fig_wc = plt.figure(figsize=(14,13))
        word_cloud = WordCloud(collocations = False, background_color = 'black', 
                            colormap = 'Wistia', min_font_size = 8,
                            stopwords=st.session_state['stopwords']).generate(text)
        plt.imshow(word_cloud, interpolation = 'bilinear')
        plt.axis("off")
        # plt.show()
        st.pyplot(fig_wc)

    if st.button("Exit"):
        st.balloons()


if __name__ == '__main__':
    main()
