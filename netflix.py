# FIRST PART. Import libries and creating the netflix dataframe 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

netflix_df = pd.read_csv('https://gist.githubusercontent.com/Ainuralmm/3bd1ebbaac091981f031ea47e7b18b61/raw/3132d3a22bdaa9edec8a639a3eae9987d25730bd/gistfile1.txt')
print(netflix_df.head(5))

# SECOND PART. DATA EXPLORATION
# showing netflix dataframe values
print(netflix_df)
# showing first 5 rows of netflix dataframe
print(netflix_df.head())
# showing last 5 rows of netflix dataframe
print(netflix_df.tail())
# showing the size of dataframe (rows, columns)
# netflix_df.shape
print(f'Data has {netflix_df.shape[1]} columns & {netflix_df.shape[0]} rows.')
# getting information like min,max and mean about numeric columns in netflix dataframe
print(netflix_df.describe(include='all'))
#same action like above but transposed
print(netflix_df.describe(include='all').T)
#To get the list of column headers in the netflix dataframe
print(netflix_df.columns)
#to clean the data set to remove a few unnecessary columns 
netflix_df.drop(['show_id','date_added','rating'], axis=1, inplace=True)
#let's return to see the result
print(netflix_df)
#so I removed unneseccery columns and now I want to rename some columns 
netflix_df.rename(columns={'type':'genre', 'title':'movie_title'}, inplace=True)
print(netflix_df.columns)
#now I want to change headings' letters to capital letters
netflix_df.columns = map(str.upper,netflix_df.columns)
print(netflix_df.columns)
# before moving to next cleaning part, I want to show how much the number of missing values in dataset
print(netflix_df.isnull().sum())


#THIRD PART. DATA CLEANING
# Cleaning "DIRECTOR" column
#so now I show how much the number of null values of 'director' column
print(netflix_df['DIRECTOR'].isnull().sum())
#here is removing rows containing null values of 'director' column 
netflix_df.dropna(subset=['DIRECTOR'], inplace=True)
#filling missing values as 'Unknown'
netflix_df['DIRECTOR'] = netflix_df['DIRECTOR'].fillna('Unknown')
#returning a final result of null values in coulmn
print(netflix_df['DIRECTOR'].isnull().sum())
#returning a final result of notnull values in column
print(netflix_df['DIRECTOR'].notnull().sum())
#checking null values with .info method
print(netflix_df.info())
#summing up null values with  .isnull method in all dataset
print(netflix_df.isnull().sum())


#Cleaning "CAST" column
#so now for cleaning "Cast" column I will do same actions like above
print(netflix_df['CAST'].isnull().sum())
#removing rows containing null values of 'director' column 
netflix_df.dropna(subset=['CAST'], inplace=True)
#filling missing values as 'Unknown'
netflix_df['CAST'] = netflix_df['CAST'].fillna('unknown')
#returning a final result of null values in column
print(netflix_df['CAST'].isnull().sum())
#checking null values with .info method
print(netflix_df.info())
#summing up null values with  .isnull method in all dataset
print(netflix_df.isnull().sum())


#Cleaning "COUNTRY" column
#so I will do same actions like above
print(netflix_df['COUNTRY'].isnull().sum())
netflix_df.dropna(subset=['COUNTRY'], inplace=True)
netflix_df['COUNTRY'] = netflix_df['COUNTRY'].fillna('unknown')
print(netflix_df['COUNTRY'].isnull().sum())
print(netflix_df['COUNTRY'].notnull().sum())
print(netflix_df.info())
print(netflix_df.isnull().sum())


#Cleaning "DURATION" column
#THe Cleaning of "DURATION" column will be little bit different. I will fill null values with mean values in duration columns
print(netflix_df['DURATION'].isnull().sum())
#as usually next I will drop null values
netflix_df.dropna(subset=['DURATION'], inplace=True)
#and now before fill with mean values i should remove all strings in "duratiom"column
netflix_df['DURATION'] = netflix_df['DURATION'].str.replace(r'\D', '')
#so finally filling with mean values
netflix_df['DURATION'].fillna(netflix_df['DURATION'].mean(), inplace=True)
#next steps are same like above
print(netflix_df['DURATION'].isnull().sum())
print(netflix_df['DURATION'].notnull().sum())
print(netflix_df.info())
print(netflix_df.isnull().sum())


#FOURTH PART. MORE EXPLORATION for Data Vizualiziotion
#to show correlation I should change a type of 'duration' column to float
netflix_df['DURATION']=np.float64(netflix_df['DURATION'])
print(netflix_df.head())
#correlation only between 2 columns, bcs other columns are strings
print(netflix_df.corr())
#number of movies and tv-shows in dataset
print(netflix_df.GENRE.value_counts())
#top 10 contries which create contets
top_10_countries = netflix_df['COUNTRY'].value_counts()[0:10]
print(top_10_countries)
#another way to show top 10 contries which create content
print(netflix_df.COUNTRY.value_counts().head(10))
#the last 10 countires which create content
print(netflix_df.COUNTRY.value_counts().tail(10))
#the amount of content was created first 15 years
print(netflix_df.RELEASE_YEAR.value_counts().head(15))
#the amount of content was created last 15 years
print(netflix_df.RELEASE_YEAR.value_counts().tail(15))
#top 10 directors who create content
top_10_directors = netflix_df['DIRECTOR'].value_counts()[0:10]
print(top_10_directors)
#the last 10 directors which create content
print(netflix_df.DIRECTOR.value_counts().tail(10))
#10 films of frequent duration in dataset
print(netflix_df.DURATION.value_counts().head(10))
#10 movies of rare duration
print(netflix_df.DURATION.value_counts().tail(10))
#the 10 most popular genres in dataset
print(netflix_df.LISTED_IN.value_counts().head(10))
#the most popular actors.but result is not correct bcs of comma in some rows
print(netflix_df.CAST.value_counts().head(10))
#so to get clear info about the 10 most popular i should to remove comma
most_popular_10_actors = pd.concat([pd.Series(i.split(',')) for i in netflix_df.CAST]).value_counts().head(10)
print(most_popular_10_actors)
#dataframe where are showed only movies
netflix_movie = netflix_df['GENRE']=='Movie'
print(netflix_df[netflix_movie])
#10 years with the highest total film releases
highest_movie_releases=netflix_df[netflix_movie]['RELEASE_YEAR'].value_counts()[0:10]
highest_movie_releases.name=None
print(highest_movie_releases)
#tv-shows dataframe 
netflix_tvshows = netflix_df['GENRE']=='TV Show'
print(netflix_df[netflix_tvshows])
#5 years with the highest total tv-shows releases
print(netflix_df[netflix_tvshows]['RELEASE_YEAR'].value_counts()[0:5])
#5 years with the lowest total tv-shows releases
print(netflix_df[netflix_tvshows]['RELEASE_YEAR'].value_counts(ascending=True)[0:5])
#10 Countries with the Most TV Shows
countries_with_the_most_tvshows = netflix_df[netflix_tvshows]['COUNTRY'].value_counts()[0:5]
print(countries_with_the_most_tvshows)
#5 countries that created the fewest TV shows
print(netflix_df[netflix_tvshows]['COUNTRY'].value_counts(ascending=True).head(5))


#FIFTH PART.DATA VISUALIZATION
#so first I will show a correlation between 2 columns
plt.figure(figsize=(10,8)) 
sns.heatmap(netflix_df.corr(), annot=True)
plt.show()

#another important to provide is number of movies and tv-shows in dataset
labels = ['Movie', 'TV-Shows']
plt.figure(figsize=(10,6))
plt.pie(netflix_df['GENRE'].value_counts(), labels=labels, autopct='%.2f%%', startangle=90, shadow=True)
plt.title('Number of Movies vs Number of TV-Shows', fontweight='bold', fontsize=14)
plt.legend()
plt.show()

#total film releases
plt.figure(figsize=(10,6))
sns.set(style="whitegrid")
sns.lineplot(data=highest_movie_releases)
plt.title('Total movie releases', fontweight='bold', fontsize=14)
plt.ylabel('Content created')
plt.xlabel('Year')
plt.show()

#5 countries which created the most TV-Shows
import pylab
countries_with_the_most_tvshows.plot.pie(figsize=(10, 8),autopct='%1.1f%%',shadow=True,explode=(0.1, 0.1, 0.1, 0.1,0.1),colors=['blue', 'green', 'yellow', 'orange','red']);
plt.title('5 countries which created the most TV-Shows', fontweight='bold', fontsize=14)
pylab.ylabel('')
plt.show()

# Duration of content in dataset
plt.figure(figsize=(10,6))
sns.set(style="whitegrid")
plt.plot(netflix_df.DURATION.value_counts().sort_index())
plt.title('Duration of content in dataset', fontweight='bold', fontsize=14)
plt.xlabel('Minutes')
plt.ylabel('Amounts')
plt.show()

#top 10 contries which create contets
plt.figure(figsize=(10,6))
plt.title('Top 10 content creating countries',fontweight='bold', fontsize=14)
x = top_10_countries.index
y = top_10_countries
plt.ylabel('Content created')
plt.bar(x, y)
plt.xticks(rotation=55)
plt.show()

#the amount of content was created first 15 years
plt.figure(figsize=(12,10))
sns.set(style="whitegrid")
ax = sns.countplot(y="RELEASE_YEAR", data=netflix_df, palette="Set2", order=netflix_df['RELEASE_YEAR'].value_counts().index[0:15])
plt.title ('Amount of content for first 15 years',fontweight='bold', fontsize=14)
plt.show()

#the amount of content was created last 15 years
plt.figure(figsize=(12,10))
sns.set(style="dark")
ax = sns.countplot(y="RELEASE_YEAR", data=netflix_df, palette="mako", order=netflix_df.RELEASE_YEAR.value_counts().index[-15:])
plt.title ('Amount of content for last 15 years',fontweight='bold', fontsize=14)
plt.show()

#the amount of popular genres in dataset
plt.figure(figsize=(12,10))
sns.set(style="dark")
ax = sns.countplot(y="LISTED_IN", data=netflix_df, palette="rocket", order=netflix_df.LISTED_IN.value_counts().index[0:10])
plt.title ('Amount of popular genres in dataset',fontweight='bold', fontsize=14)
plt.show()

#top 10 directors who create content
plt.figure(figsize=(10,6))
plt.title('Top 10 directors who create contents',fontweight='bold', fontsize=14)
x = top_10_directors.index
y = top_10_directors
plt.ylabel('Content created')
plt.bar(x, y)
plt.xticks(rotation=45)
plt.show()

#same result but another design of plot
plt.figure(figsize=(12,10))
sns.set(style="darkgrid")
ax = sns.countplot(y="DIRECTOR", data=netflix_df, palette="husl", order=netflix_df['DIRECTOR'].value_counts().index[0:10])
plt.title('Top 10 directors who create contents',fontweight='bold', fontsize=14)
plt.show()

#top 10 actors 
plt.figure(figsize=(10,6))
sns.set(style="darkgrid")
plt.title('Top 10 actors', fontweight='bold', fontsize=14)
x = most_popular_10_actors.index
y = most_popular_10_actors
plt.ylabel('Content created',fontweight='bold', fontsize=14)
plt.bar(x, y, color=['purple','violet'])
plt.xticks(rotation=45)
plt.show()


#SIXTH PART.MODELING
# importing required tools from sklearn library
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

#making copy of dataset
netflix_df1=netflix_df.copy()

# defining function for changing string values to numeric values for better modelling
def encode_df(dataframe):
    le = LabelEncoder()
    for column in dataframe.columns:
        dataframe[column] = le.fit_transform(dataframe[column])
    return dataframe

#this is our result
print(encode_df(netflix_df1))

# defining prediction values/i want to predict how much content will be every year 
y_country = netflix_df1['RELEASE_YEAR']
x_country = netflix_df1.drop(['RELEASE_YEAR'], axis=1)

# defining train and test data
x_train, x_test, y_train, y_test = train_test_split(x_country, y_country, test_size=0.2, random_state=42)
# defining model and training data
model = GaussianNB()
model.fit(x_train, y_train)
# prediction on test data
y_pred = model.predict(x_test)
# Calculating accuracy score
accuracy_score(y_test, y_pred)
print(sum(y_pred == y_test) / len(y_test))

model = RandomForestClassifier()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy_score(y_test, y_pred)


#Classification
#classification of every column to each other
sns.pairplot(netflix_df1, height=1.5)

#Practical Clustering
#to make a clear clustering, i want to take only 100 rows from dataset
netflix_df2=netflix_df1.head(100)
#showing every countries and their movies/tv-shows usual duration
#here between 0 and 25 are tv-shows
plt.figure(figsize=(10,6))
plt.scatter(netflix_df2['DURATION'], netflix_df2['COUNTRY'])
plt.xlabel('DURATION')
plt.ylabel('COUNTRY')
plt.show()