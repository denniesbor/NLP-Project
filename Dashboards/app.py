import streamlit as st
import warnings
import os
import json
import re
from PIL import Image
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


# import database database
import sqlalchemy as db
from sqlalchemy import Table, Column, INTEGER, MetaData
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

#load secret files
from dotenv import load_dotenv
load_dotenv()

# style
STYLE = """
<style>
img {
    max-width: 100%;
}
</style> """

# configure db info
USER = os.getenv("user")
PASSWORD = os.getenv("password")
HOST = os.getenv("host")
PORT = 3306
DATABASE = "nlp"
TABLENAME = "us_house_reps"

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

try:
  df = pd.read_csv('all_tweets.csv')
except FileNotFoundError:
  
    # convert to a dataframe
    df = pd.DataFrame(query)
    df.columns = query[0].keys()

    # remove empty tweets
    df = df[df.text != '']

    # drop duplicates
    df = df.drop_duplicates(subset = ['text'])
    df.reset_index(inplace=True, drop=False)
    df.to_csv('all_tweets.csv', index=False)

def main():

    html_temp = """
	<div style="background-color:green;"><p style="color:white;font-size:40px;padding:9px">Ruto vs Raila Tweets</p></div>
	"""
    st.markdown(html_temp, unsafe_allow_html=True)

    image = Image.open('Logo1.jpg')
    st.image(image, caption='Twitter for Analytics',use_column_width=True)
    
    # load the dataframe
    try:
        df = pd.read_csv('all_tweets.csv')
    except FileNotFoundError:
  
        # convert to query results to a dataframe
        df = pd.DataFrame(query)
        df.columns = query[0].keys()

        # remove empty tweets
        df = df[df.text != '']

        # drop duplicates
        df = df.drop_duplicates(subset = ['text'])
        df.reset_index(inplace=True, drop=False)
        df.to_csv('all_tweets.csv', index=False)
    
    df = pd.read_csv('rutoraila.csv')
    
    politician = st.radio(
     "Select Politician",
     ('williamsruto', 'railaodinga', 'all'))
    
    
    if politician == 'williamsruto' or politician == 'railaodinga':
        
        with st.spinner(f"Please wait, extracting {politician} tweets"):
            pol = politician
            df_ind = df.query('username==@pol')

            fig, axs = plt.subplots(2,2,figsize=(10,7))
            fig.tight_layout()
            # axis 0,0 pie chart

            a=len(df_ind[df_ind["sentiment"]=="positive"])
            b=len(df_ind[df_ind["sentiment"]=="negative"])
            c=len(df_ind[df_ind["sentiment"]=="neutral"])
            d=np.array([a,b,c])
            explode = (0.1, 0.0, 0.1)
            axs[0,0].pie(d,shadow=True,explode=explode,labels=["Positive","Negative","Neutral"],autopct='%1.2f%%')


            # axis 0,1 wc

            df_neg = df_ind[df_ind['sentiment']=='neutral']
            all_words = ' '.join(twts for twts in df_neg['text'])

            text_cloud = WordCloud(height=300,width=500,random_state=10,max_font_size=110).generate(all_words)
            axs[0,1].set_title('Neutral Sentiments')
            axs[0,1].imshow(text_cloud,interpolation='bilinear')
            axs[0,1].axis('off')


            # axis 0,1 wc


            df_neg = df_ind[df_ind['sentiment']=='positive']
            all_words = ' '.join(twts for twts in df_neg['text'])

            text_cloud = WordCloud(height=300,width=500,random_state=10,max_font_size=110).generate(all_words)

            axs[1,0].set_title('Positive Sentiments')
            axs[1,0].imshow(text_cloud,interpolation='bilinear')
            axs[1,0].axis('off')


            # axis 0,1 wc

            all_words = ' '.join(twts for twts in df_ind['text'])

            text_cloud = WordCloud(height=300,width=500,random_state=10,max_font_size=110).generate(all_words)

            axs[1,1].set_title('All Words')
            axs[1,1].imshow(text_cloud,interpolation='bilinear')
            axs[1,1].axis('off')
            st.pyplot(fig)
        st.success(f'Sucessfully plotted {politician} Tweets !!!!')   
    else:
        st.write("You selected all")

    
               
    st.sidebar.header("About App")
    st.sidebar.info("Summer internship project")
    

        
        
    # st.sidebar.header("For Any Queries/Suggestions Please reach out at :")
    # st.sidebar.info("darekarabhishek@gmail.com")
    #st.sidebar.subheader("Scatter-plot setup")
    #box1 = st.sidebar.selectbox(label= "X axis", options = numeric_columns)
    #box2 = st.sidebar.selectbox(label="Y axis", options=numeric_columns)
    #sns.jointplot(x=box1, y= box2, data=df, kind = "reg", color= "red")
    #st.pyplot()



    if st.button("Exit"):
        st.balloons()



if __name__ == '__main__':
    main()
