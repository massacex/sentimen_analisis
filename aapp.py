import streamlit as st 
import pandas as pd 
import csv
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize
import string
import re
import numpy as np 
import seaborn as sns 
from nltk.corpus import stopwords 
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report


st.title('sentimen analisis')
st.write('silahkan masukkan file anda')

upload_file = st.file_uploader("upload your csv")
if upload_file is not None:
	input_file=pd.read_csv(upload_file)
	st.write(input_file)

	#pelabelan lexicon
	lexicon = dict()
	pos_lexicon = pd.read_csv('positive - positive.csv',sep='\t')
	neg_lexicon = pd.read_csv('negative - negative.csv',sep='\t')

	lexicon = pd.read_csv('modified_full_lexicon.csv')
	lexicon = lexicon.reset_index(drop=True)

	negasi = ['bukan','tidak','ga','gk']
	lexicon_word = lexicon['word'].to_list()
	lexicon_num_words = lexicon['number_of_words']

	sencol =[]
	senrow =np.array([])
	nsen = 0
	factory = StemmerFactory()
	stemmer = factory.create_stemmer()
	sentiment_list = []
	# berfungsi untuk menulis sentimen kata
	def found_word(ind,words,word,sen,sencol,sentiment,add):
	    # jika sudah termasuk dalam bag of words matrix, maka tinggal menambah nilainya
	    if word in sencol:
	        sen[sencol.index(word)] += 1
	    else:
	    #jika tidak, menambahkan kata baru
	        sencol.append(word)
	        sen.append(1)
	        add += 1
	    if (words[ind-1] in negasi):
	        sentiment += -lexicon['weight'][lexicon_word.index(word)]
	    else:
	        sentiment += lexicon['weight'][lexicon_word.index(word)]
	    
	    return sen,sencol,sentiment,add
	            
	# memeriksa setiap kata, jika mereka muncul dalam leksikon, dan kemudian menghitung sentimen mereka jika mereka muncul
	for i in range(len(input_file)):
	    nsen = senrow.shape[0]
	    words = word_tokenize(input_file["review"][i])
	    sentiment = 0 
	    add = 0
	    prev = [0 for ii in range(len(words))]
	    n_words = len(words)
	    if len(sencol)>0:
	        sen =[0 for j in range(len(sencol))]
	    else:
	        sen =[]
	    
	    for word in words:
	        ind = words.index(word)
	# periksa apakah mereka termasuk dalam leksikon
	        if word in lexicon_word :
	            sen,sencol,sentiment,add= found_word(ind,words,word,sen,sencol,sentiment,add)
	        else:
	        # if not, then check the root word
	            kata_dasar = stemmer.stem(word)
	            if kata_dasar in lexicon_word:
	                sen,sencol,sentiment,add= found_word(ind,words,kata_dasar,sen,sencol,sentiment,add)
	        # jika masih negatif, coba cocokkan kombinasi kata dengan kata yang berdekatan
	            elif(n_words>1):
	                if ind-1>-1:
	                    back_1    = words[ind-1]+' '+word
	                    if (back_1 in lexicon_word):
	                        sen,sencol,sentiment,add= found_word(ind,words,back_1,sen,sencol,sentiment,add)
	                    elif(ind-2>-1):
	                        back_2    = words[ind-2]+' '+back_1
	                        if back_2 in lexicon_word:
	                            sen,sencol,sentiment,add= found_word(ind,words,back_2,sen,sencol,sentiment,add)
	    if add>0:  
	        if i>0:
	            if (nsen==0):
	                senrow = np.zeros([i,add],dtype=int)
	            elif(i!=nsen):
	                padding_h = np.zeros([nsen,add],dtype=int)
	                senrow = np.hstack((senrow,padding_h))
	                padding_v = np.zeros([(i-nsen),senrow.shape[1]],dtype=int)
	                senrow = np.vstack((senrow,padding_v))
	            else:
	                padding =np.zeros([nsen,add],dtype=int)
	                senrow = np.hstack((senrow,padding))
	            senrow = np.vstack((senrow,sen))
	        if i==0:
	            senrow = np.array(sen).reshape(1,len(sen))
	    # jika tidak ada maka perbarui saja matriks lama
	    elif(nsen>0):
	        senrow = np.vstack((senrow,sen))
	        
	    sentiment_list.append(sentiment)

	sencol.append('sentiment')
	sentiment_array = np.array(sentiment_list).reshape(senrow.shape[0],1)
	sentiment_data = np.hstack((senrow,sentiment_array))
	df_sen = pd.DataFrame(sentiment_data,columns = sencol)

	cek_df = pd.DataFrame([])
	cek_df['text'] = input_file["review"].copy()
	cek_df['sentiment']  = df_sen['sentiment'].copy()

	result = []
	for sentimen in cek_df['sentiment']:
	    if(sentimen>0):
	        result.append('1')
	    else:
	        result.append('0')
	        
	cek_df['hasil'] = result
	st.write(result)

	st.title("tampilan grafik")
	diagram = sns.countplot(cek_df['hasil'])
	st.pyplot(diagram.figure)

	#proses preprocessing
	def remove_pattern(input_txt, pattern):
	    r = re.findall(pattern, input_txt)
	    for i in r:
	        input_txt = re.sub(i, '', input_txt)
	    return input_txt    
	cek_df['hapus_user'] = np.vectorize(remove_pattern)(cek_df['text'], "@[\w]*")
	

	#hapus http
	def remove(tweet):
	    #menghapus angka
	    tweet = re.sub('[0-9]+', '', tweet)
	    
	    #menghapus $GE
	    tweet = re.sub(r'\$\w*', '', tweet)
	 
	    #menghapus text "RT"
	    tweet = re.sub(r'^RT[\s]+', '', tweet)
	    
	    #hanya menghapus tanda hashtag
	    tweet = re.sub(r'#', '', tweet)
	    return tweet
	cek_df['remove_http'] = cek_df['hapus_user'].apply(lambda x: remove(x))
	cek_df
	cek_df.sort_values("remove_http", inplace = True)
	

	#import stopword
	stopwords_indonesia = stopwords.words('indonesian')
	factory = StemmerFactory()
	stemmer = factory.create_stemmer()

	#menghapus emoticon bahagia
	emoticons_happy = set([
	    ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
	    ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
	    '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
	    'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
	    '<3'
	    ])
	 
	#menghapus emoticon sedih
	emoticons_sad = set([
	    ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
	    ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
	    ':c', ':{', '>:\\', ';('
	    ])
	 
	#gabungan semua emotiocn
	emoticons = emoticons_happy.union(emoticons_sad)
	cek_df

	def clean_tweets(tweet):
    # remove stock market tickers like $GE
	    tweet = re.sub(r'\$\w*', '', tweet)
	 
	    # remove old style retweet text "RT"
	    tweet = re.sub(r'^RT[\s]+', '', tweet)
	 
	    # remove hyperlinks
	    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
	    
	    # remove hashtags
	    # only removing the hash # sign from the word
	    tweet = re.sub(r'#', '', tweet)
	    
	    #remove coma
	    tweet = re.sub(r',','',tweet)
	    
	    #remove angka
	    tweet = re.sub('[0-9]+', '', tweet)
	 
	    # tokenize tweets
	    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
	    tweet_tokens = tokenizer.tokenize(tweet)
	 
	    tweets_clean = []    
	    for word in tweet_tokens:
	        if (word not in stopwords_indonesia and # remove stopwords
	              word not in emoticons and # remove emoticons
	                word not in string.punctuation): # remove punctuation
	            #tweets_clean.append(word)
	            stem_word = stemmer.stem(word) # stemming word
	            tweets_clean.append(stem_word)
	 
	    return tweets_clean
	cek_df['Review_clean'] = cek_df['remove_http'].apply(lambda x: clean_tweets(x))
	cek_df

	def remove_punct(text):
	    text  = " ".join([char for char in text if char not in string.punctuation])
	    return text
	cek_df['Review1'] = cek_df['Review_clean'].apply(lambda x: remove_punct(x))
	cek_df

	cek_df.sort_values(by='Review1', inplace=True)

#menghapus kolom yang tidak dibutuhkan : tweet, remover_user, remove_http, tweet_clean

	cek_df.drop(cek_df.columns[[0,1,3,4,5]], axis = 1, inplace = True)
	st.write(cek_df)

	#ektraksi fitur
	X, y = cek_df['Review1'], cek_df['hasil']

	#Split data in train and test sets.
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
	my_categories=['0','1']
	tfidf = CountVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 1))

	#proses klasifikasi
	# TF-IDF BASED FEATURE REPRESENTATION
	tfidf.fit_transform(X_train)
	        
	train_feature_set=tfidf.transform(X_train)
	test_feature_set=tfidf.transform(X_test)

	# instantiate the model (using the default parameters)
	NB = MultinomialNB()

	NB.fit(train_feature_set,y_train)


	#
	y_pred=NB.predict(test_feature_set)
	lrm1 = accuracy_score(y_pred, y_test)
	cm=confusion_matrix(y_test, y_pred)
	pr=precision_score(y_test, y_pred, average='micro')
	rc=recall_score(y_test, y_pred, average='micro')
	f=f1_score(y_test, y_pred, average='micro')

	st.write('accuracy %s' % lrm1)
	st.write('Precision %s' % pr)
	st.write('Recall %s' % rc)
	st.write('F1 Score %s' % rc)
	st.write(classification_report(y_test, y_pred,target_names=my_categories))
	st.write('Convolution matrix ')
	st.write(cm)

else:
	st.write("silahkan masukkan data")

#prediksi kata
kalimat = st.text_input("masukkan kata atau komentar")
if st.button("proses"):
	lower_case = kalimat.lower()
	st.write(lower_case)
	# st.write(teks)

	hasill = re.sub(r"\d+", "", lower_case)
	st.write(hasill)

	hasil = hasill.translate(str.maketrans("","",string.punctuation))
	st.write(hasil)

	hasil1 = hasil.strip()
	st.write(hasil1)

	pisah = hasil1.split()
	st.write(pisah)
else:
	st.write("masukkan komentar!!!")
