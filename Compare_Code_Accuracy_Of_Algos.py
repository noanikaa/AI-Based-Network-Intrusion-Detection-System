import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score 
from sklearn.preprocessing import MinMaxScaler 
from sklearn.manifold import TSNE 
import matplotlib.pyplot as plt 
import seaborn as sns 
from keras.layers import Input, Dense 
from keras.models import Model, Sequential 
from keras import regularizers
import pandas as pd
import numpy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, zero_one_loss
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import metrics

Index=(['normal.', 'buffer_overflow.', 'loadmodule.', 'perl.', 'neptune.',
       'smurf.', 'guess_passwd.', 'pod.', 'teardrop.', 'portsweep.',
       'ipsweep.', 'land.', 'ftp_write.', 'back.', 'imap.', 'satan.', 'phf.',
       'nmap.', 'multihop.', 'warezmaster.', 'warezclient.', 'spy.',
       'rootkit.'])

print(np.shape(Index))

print(Index[0])


# DATASET

# ---------------------------------------------------------------------------
# Must declare data_dir as the directory of training and test files
#data_dir="./datasets/KDD-CUP-99/"
raw_data_filename ="kddcup.data_10_percent_corrected"
#raw_data_filename = data_dir + "kddcup.data_10_percent"
print ("Loading raw data")
raw_data = pd.read_csv(raw_data_filename, header=None)
print ("Transforming data")
# Categorize columns: "protocol", "service", "flag", "attack_type"
raw_data[1], protocols= pd.factorize(raw_data[1])
raw_data[2], services = pd.factorize(raw_data[2])
raw_data[3], flags    = pd.factorize(raw_data[3])
raw_data[41], attacks = pd.factorize(raw_data[41])

print(attacks)
# separate features (columns 1..40) and label (column 41)
features= raw_data.iloc[:,:raw_data.shape[1]-1]
labels= raw_data.iloc[:,raw_data.shape[1]-1:]
labels= labels.values.ravel() # this becomes a 'horizontal' array
print(labels)
import matplotlib.pyplot as plt


# In[2]:



df= pd.DataFrame(features)
X, X_test, y, y_test = train_test_split(df, labels, train_size=0.5, test_size=0.1)
X_normal_scaled=X
y_normal=y
# In[13]:


clf=RandomForestClassifier(n_estimators=100)
#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_normal_scaled,y_normal)
y_pred=clf.predict(X_test)
#


# In[35]:

print('-------------------------------------------------------------\n')
print("Traditional RF Accuracy:",metrics.accuracy_score(y_test,y_pred))
print('-------------------------------------------------------------\n')


from sklearn.neighbors import KNeighborsClassifier
KNN_Classifier = KNeighborsClassifier(n_jobs=-1)
KNN_Classifier.fit(X_normal_scaled,y_normal)
y_pred=KNN_Classifier.predict(X_test)
print('-------------------------------------------------------------\n')
print("KNN Accuracy:",metrics.accuracy_score(y_test,y_pred))
print('-------------------------------------------------------------\n')


# In[9]:
