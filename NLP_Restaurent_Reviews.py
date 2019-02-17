import pandas as pd
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t',quoting=3)
#print(dataset.values)

#cleaning process
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[]
for i in range(0,1000):
    reviews=re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    reviews=reviews.lower()
    reviews=reviews.split()
    ps=PorterStemmer()
    reviews=[ps.stem(word) for word in reviews if not word in set(stopwords.words('english'))]
    reviews=' '.join(reviews)
    corpus.append(reviews)
print(corpus)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values


# Splitting the dataset into the Training set and Test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)

print((27+46)/100)
