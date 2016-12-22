#The objective is to classify symptoms of breast cancer as benign or malignant

##PART 1

#import pandas
import pandas as pd
#read table
df = pd.read_table(r"C:\Users\Alcatraz\Documents\breast-cancer-wisconsin.DATA",sep=",",header=None)
#set column names according to the source dataset
df.columns.values
df.rename(columns={0:'sample', 1:'thickness', 2:'size', 3:'shape', 4:'adhesion', 5:'epithelial', 6:'nuclei', 7:'chromatin', 8:'nucleoli', 9:'mitoses', 10:'status'},inplace=True)
#have a look
df.head()
#copy the status column
y=pd.Series(df['status'].copy())
y.head()
y.unique()
#in the previous line, 2 means benign, 4 means malignant

#remove the status column from the dataset together with 'sample' (the latter is not useful)
df.drop(labels=['status'],axis=1,inplace=True)
df.drop(labels=['sample'],axis=1,inplace=True)

#check NaN's
df.isnull().sum()
#check types
df.dtypes
#in the previous line, the variable 'nuclei' is numeric but the type is object, so convert it to the appropriate type
df['nuclei'].unique()
df['nuclei']=df['nuclei'].convert_objects(convert_numeric=True)
#in the previous line, a deprecated warning shows up but no problem, now the types are correct

#check NaN's again
df.isnull().sum()
#replace the existing NaN's with the mean of the corresponding column
df=df.fillna(df.mean())
#reset index
df=df.reset_index(drop=True)
#have a look
df.head()
#normalize my data because it the variables consist of mixed units, then convert to dataframe

from sklearn import preprocessing
df = pd.DataFrame(preprocessing.normalize(df))
#have a look, notice that the column names are gone
df.head()

#train_test_split using df and y, test_size=33% and random_state=7 for reproduceability

from sklearn.cross_validation import train_test_split
data_train,data_test,labels_train,labels_test=train_test_split(df,y,test_size=0.33,random_state=7)


#PART 2

#I will apply PCA for dimensionality reduction and the K-nearest neighbors to classify my patients

  from sklearn.decomposition import PCA  
  #reduce to 2 dimensions
  model=PCA(n_components=2)
  #train my data
  model.fit(data_train)
  #obtain the reduced data
  T=model.transform(data_train)
  #compare shapes   
  data_train.shape
  T.shape

#In order to apply K-nearest neighbors, I will change the values of the response variable of labels_train to '0' and '1'
labels_train.unique()
#encode
here={2:'0',4:'1'}
#use .map()
labels_train=labels_train.map(here)

#Now I apply K-nearest neighbors with 7 neighbors, the idea is to vary k=1 to say 15, and try the parameter 'weights' equal to 'uniform' and 'distance' 
from sklearn.neighbors import KNeighborsClassifier
model1 = KNeighborsClassifier(n_neighbors=7)
#fit with the transformed data T and the response variable labels_train
model1.fit(T,labels_train)
#obtain score for the test data
X=model.transform(data_test)
y=labels_test.map({2:'0',4:'1'})
model1.score(X,y)
#the score is 0.835

#Now I will plot the decision boundary and my classification

  import matplotlib.pyplot as plt
  import matplotlib
  matplotlib.style.use('ggplot') 
  #definitions
  fig = plt.figure()
  ax = fig.add_subplot(111)
  #padding and resolution
  padding = 0.1
  resolution = 0.1

  #define colors, 0 for benign, 1 for malignant
  colors = {0:'royalblue',1:'lightsalmon'} 
 
  #calculate the boundaris
  x_min, x_max = T[:, 0].min(), T[:, 0].max()
  y_min, y_max = T[:, 1].min(), T[:, 1].max()
  x_range = x_max - x_min
  y_range = y_max - y_min
  x_min -= x_range * padding
  y_min -= y_range * padding
  x_max += x_range * padding
  y_max += y_range * padding

  #Now create a 2D grid matrix. The values stored in the matrix are the predictions of the class at the corresponding location
  import numpy as np
  xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                       np.arange(y_min, y_max, resolution))

  #what class does my classifier model predict?
  Z = model1.predict(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)

  #plot the contour map
  plt.contourf(xx, yy,Z, cmap=plt.cm.seismic)
  plt.axis('tight')

  #plot my samples and their classification
  for label in np.unique(labels_train):
    indices = np.where(labels_train == label)
    plt.scatter(T[indices, 0], T[indices, 1],marker="o", c=colors[pd.to_numeric(label)], alpha=0.8)

  p = model1.get_params()
  plt.title('K = ' + str(p['n_neighbors']))
  plt.show()


#PART 3

#Now I repeat the same procedure as in PART 2 but using Isomap instead of PCA

  from sklearn import manifold
  #I use 5 neighbors but I should try between 5 and 10. The dimension should be two as in PART 2
  model3 = manifold.Isomap(n_neighbors=5,n_components=2)
  #fit data
  model3.fit(data_train)
  #transform  
  manifold = model3.transform(data_train)
  #compare shapes  
  data_train.shape
  manifold.shape
  
#Repeat the k-nearest neighbors procedure and then the plot procedure
  
  #Now I apply K-nearest neighbors with 7 neighbors, the idea is to vary k=1 to say 15, and try the parameter 'weights' equal to 'uniform' and 'distance' 
from sklearn.neighbors import KNeighborsClassifier
model4 = KNeighborsClassifier(n_neighbors=7)
#fit with the transformed data 'manifold' and the response variable labels_train
model4.fit(manifold,labels_train)

#calculate score of my test data
X=model3.transform(data_test)
y=labels_test.map({2:'0',4:'1'})
model4.score(X,y)
#the score is 0.883

#Now I will plot the decision boundary and my classification

  import matplotlib.pyplot as plt
  import matplotlib
  matplotlib.style.use('ggplot') 
  #definitions
  fig = plt.figure()
  ax = fig.add_subplot(111)
  #padding and resolution
  padding = 0.1
  resolution = 0.1

  #define colors, 0 for benign, 1 for malignant
  colors = {0:'royalblue',1:'lightsalmon'} 
 
  #calculate the boundaris
  x_min, x_max = manifold[:, 0].min(), manifold[:, 0].max()
  y_min, y_max = manifold[:, 1].min(), manifold[:, 1].max()
  x_range = x_max - x_min
  y_range = y_max - y_min
  x_min -= x_range * padding
  y_min -= y_range * padding
  x_max += x_range * padding
  y_max += y_range * padding

  #Now create a 2D grid matrix. The values stored in the matrix are the predictions of the class at the corresponding location
  import numpy as np
  xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                       np.arange(y_min, y_max, resolution))

  #what class does my classifier model predict?
  Z = model4.predict(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)

  #plot the contour map
  plt.contourf(xx, yy,Z, cmap=plt.cm.seismic)
  plt.axis('tight')

  #plot my samples and their classification
  for label in np.unique(labels_train):
    indices = np.where(labels_train == label)
    plt.scatter(manifold[indices, 0], manifold[indices, 1],marker="o", c=colors[pd.to_numeric(label)], alpha=0.8)

  p = model4.get_params()
  plt.title('K = ' + str(p['n_neighbors']))
  plt.show()




