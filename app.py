import streamlit as st
import warnings


import pandas as pd
import numpy as np
import pandas as pd
import tweepy
import json
from tweepy import OAuthHandler
import re
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

warnings.filterwarnings("ignore")

#To Hide Warnings
st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_option('deprecation.showPyplotGlobalUse', False)
# Viz Pkgs
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
#sns.set_style('darkgrid')


STYLE = """
<style>
img {
    max-width: 100%;
}
</style> """

def main():

    html_temp = """
	<div style="background-color:green;"><p style="color:white;font-size:40px;padding:9px">Ruto vs Raila Tweets</p></div>
	"""
    st.markdown(html_temp, unsafe_allow_html=True)

    from PIL import Image
    image = Image.open('Logo1.jpg')
    st.image(image, caption='Twitter for Analytics',use_column_width=True)
    
    
    # load the dataframe
    
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
