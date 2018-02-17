import pandas as pd 
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from textblob import TextBlob
      
data = pd.read_excel("uber_final.xlsx") #change the file names

reviews = []
for review in data["review"]:
    review = re.sub("[^a-zA-Z]", " ", review) 
    words = review.lower().split()                             
    stops = set(stopwords.words("english"))                  
    words = [w for w in words if not w in stops]
    review = "".join(words)                                                             
    reviews.append(review)


tf = TfidfVectorizer(analyzer='word',ngram_range=(1,1), min_df = 1, stop_words = 'english')
features = tf.fit_transform(reviews).toarray()
df=pd.DataFrame(features)

x_train, x_test, y_train, y_test = train_test_split(df, data['rating'], test_size=0.25,random_state=3)

l=LogisticRegression(multi_class='multinomial',solver ='newton-cg').fit(x_train,y_train)
p_rating=l.predict(x_test)
print("Accuracy of the Prediction is")
print(accuracy_score(y_test,p_rating))

spam=0
for i in range(0,len(data)):
	blob = TextBlob(data.iloc[i]['review'])
	if blob.sentiment.polarity>0:
		if data.iloc[i]['rating']<4:
			spam=spam+1
	elif blob.sentiment.polarity<0:
		if data.iloc[i]['rating']>=4:
			spam=spam+1

print("No. of spam/inconsistent reviews is")
print(spam)

