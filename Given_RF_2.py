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
plt.plot(labels);
plt.show()


df= pd.DataFrame(features)
X, X_test, y, y_test = train_test_split(df, labels, train_size=0.8, test_size=0.2)

print ("X, y :", X.shape, y.shape)
print ("X_test, y_test:", X_test.shape, y_test.shape)

# Scaling the data to make it suitable for the auto-encoder 
X_scaled = MinMaxScaler().fit_transform(X) 
X_normal_scaled = X_scaled[y == 0] 
X_ids_scaled = X_scaled[y == 1] 

# Building the Input Layer 
input_layer = Input(shape =(X.shape[1], )) 

# Building the Encoder network 
encoded = Dense(100, activation ='tanh',activity_regularizer = regularizers.l1(10e-5))(input_layer) 
encoded = Dense(50, activation ='tanh',activity_regularizer = regularizers.l1(10e-5))(encoded) 
encoded = Dense(25, activation ='tanh',activity_regularizer = regularizers.l1(10e-5))(encoded) 
encoded = Dense(12, activation ='tanh',activity_regularizer = regularizers.l1(10e-5))(encoded) 
encoded = Dense(6, activation ='relu')(encoded) 

# Building the Decoder network 
decoded = Dense(12, activation ='tanh')(encoded) 
decoded = Dense(25, activation ='tanh')(decoded) 
decoded = Dense(50, activation ='tanh')(decoded) 
decoded = Dense(100, activation ='tanh')(decoded) 

# Building the Output Layer 
output_layer = Dense(X.shape[1], activation ='relu')(decoded) 

# Defining the parameters of the Auto-encoder network 
autoencoder = Model(input_layer, output_layer) 
autoencoder.compile(optimizer ="adadelta", loss ="mse") 

# Training the Auto-encoder network 
autoencoder.fit(X_normal_scaled, X_normal_scaled,batch_size = 16, epochs =1,shuffle = True, validation_split = 0.20) 

hidden_representation = Sequential() 
hidden_representation.add(autoencoder.layers[0]) 
hidden_representation.add(autoencoder.layers[1]) 
hidden_representation.add(autoencoder.layers[2]) 
hidden_representation.add(autoencoder.layers[3]) 
hidden_representation.add(autoencoder.layers[4])

# Separating the points encoded by the Auto-encoder as normal and ids 
normal_hidden_rep = hidden_representation.predict(X_normal_scaled) 
ids_hidden_rep = hidden_representation.predict(X_ids_scaled) 

# Combining the encoded points into a single table 
encoded_X = np.append(normal_hidden_rep, ids_hidden_rep, axis = 0) 
y_normal = np.zeros(normal_hidden_rep.shape[0]) 
y_ids = np.ones(ids_hidden_rep.shape[0]) 
encoded_y = np.append(y_normal, y_ids) 

print(np.shape(encoded_X))
print(np.shape(encoded_y))
#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)
#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(encoded_X,encoded_y)
y_pred=clf.predict(encoded_X)
print("Accuracy:",metrics.accuracy_score(encoded_y, y_pred))

for ik in range(len(y_pred)):
    print(Index[y_pred[ik]])
