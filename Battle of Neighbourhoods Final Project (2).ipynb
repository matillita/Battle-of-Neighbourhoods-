#!/usr/bin/env python
# coding: utf-8

# 
# # 1. Introduction/Business Problem
# 
# I am from Malaga Spain, specifically from a neighborhood called Pedregalejo,a beautiful spot for EU Nordic students who come to Malaga to learn spanish in the cost of Andalusia. 
# 
# The origin of the name of this area is unclear as Pedregalejo comes from the word pedregal meaning quarry. However, it was called this way even before a quarry even existed there. there is a lot of fishing tradition. In fact, many malagueños head to Pedregalejo on weekends to have lunch in one of the many fish restaurants on the boardwalk. There are still a few family-owned restaurants that head out to sea early in the morning and then sell in their restaurants the “catch of the day”. 
# 
# ## Opening of Student Residence
# 
# The challenge is to find a suitable location for opening a new Student Residence nearby the beach and language schools where are  high variety of restaurants and hotels.
# 
# ## Expected / Interested Audience
# 
# Students would come from wealthy countries such as Norway, Finland, Sweden or Denmark. They are really interested in the Spanish culture and food.
# 
# 
# 
# ![image.png](attachment:image.png)
# 
# 

# 
# # 2.Data section
# 
# ## 2.a What data is used?
# 
# We will be completely working on Foursquare data to explore and try to locate our residence where more student residences, Language centers with Spanish courses, beaches, restaurants are present nearby.
# 
# How will we be solving using this data?
# 
# We will looking for midpoint area of venues to locate our new hotel.Before that our major focus will be on all venues present in and around the core place of Pedregalejo.

# ## 2.b Importing Libraries

# In[1]:


get_ipython().system('pip install beautifulsoup4')
get_ipython().system('pip install lxml')
from bs4 import BeautifulSoup
import requests
import pandas as pd


# In[2]:


# Import libraries
import numpy as np # data in a vectorized manner manipulation
import pandas as pd # data analsysis
import requests # HTTP library
from bs4 import BeautifulSoup # scraping library

from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe
import json # JSON files manipulation

from sklearn.cluster import KMeans # clustering algorithm

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt

get_ipython().system('conda install -c conda-forge geopy --yes ')
from geopy.geocoders import Nominatim

get_ipython().system('conda install -c conda-forge folium=0.5.0 --yes')
import folium # map rendering library

print("*********   Loaded library     ***********")


# ## 2.c Credentials and Core location

# In[3]:


CLIENT_ID = 'WZHLRHG4IS3IBOBVAE4GCB24AWAGFB1H2SUK0EH454KUKT32' # your Foursquare ID
CLIENT_SECRET = 'AJZ5ZYBVLBAFZ2E4XHE5RDYIU0KHV5VGGFIS35H2COWTHC40' # your Foursquare Secret
VERSION = '20180604'
LIMIT = 150

address = "Pedregalejo, Malaga"

geolocator = Nominatim(user_agent="foursquare_agent")
location = geolocator.geocode(address)
latitude =location.latitude   #8.079252 # location.latitude 
longitude =location.longitude #77.5499338 # location.longitude # 

marb='Pedregalejo location : {},{}'.format(latitude,longitude)
print(marb)


# ## 2.d Search for hotel, language schools and restaurant within 500M

# In[4]:



#Quering for hotel & restaurant

search_query_hot = 'hotel'
search_query_SC = 'language school'
search_query_SR = 'restaurant'

radius = 500
url_hotel = 'https://api.foursquare.com/v2/venues/search?client_id={}&client_secret={}&ll={},{}&v={}&query={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET, latitude, longitude, VERSION, search_query_hot, radius, LIMIT)
url_SC = 'https://api.foursquare.com/v2/venues/search?client_id={}&client_secret={}&ll={},{}&v={}&query={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET, latitude, longitude, VERSION, search_query_SC, radius, LIMIT)
url_SR = 'https://api.foursquare.com/v2/venues/search?client_id={}&client_secret={}&ll={},{}&v={}&query={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET, latitude, longitude, VERSION, search_query_SR, radius, LIMIT)
#url


# ### Send the GET Request of hotel & restaurants and examine the results

# In[5]:


results_hotel = requests.get(url_hotel).json()
results_SC = requests.get(url_SC).json()
results_SR = requests.get(url_SR).json()
#results_hotel


# ### Get relevant part of JSON and transform it into a pandas dataframe

# In[7]:


# assign relevant part of JSON to venues
venues_hotel = results_hotel['response']['venues']
venues_SC = results_SC['response']['venues']
venues_SR = results_SR['response']['venues']

# tranform venues into a dataframe and merging both data
dataframe_hotel = json_normalize(venues_hotel)
dataframe_SC = json_normalize(venues_SC)
dataframe_SR = json_normalize(venues_SR)

dataframe = pd.concat([dataframe_hotel,dataframe_SC,dataframe_SR])

print("There are {} hotels, language schools and Restaurants at Pedregalejo".format(dataframe.shape[0]))


# ### Define information of interest and filter dataframe

# In[8]:


# keep only columns that include venue name, and anything that is associated with location
filtered_columns = ['name', 'categories'] + [col for col in dataframe.columns if col.startswith('location.')] + ['id']
dataframe_filtered = dataframe.loc[:, filtered_columns]

# function that extracts the category of the venue

def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']

    
# filter the category for each row
dataframe_filtered['categories'] = dataframe_filtered.apply(get_category_type, axis=1)

  
# clean column names by keeping only last term
dataframe_filtered.columns = [column.split('.')[-1] for column in dataframe_filtered.columns]

#dataframe_filtered
hotels_df=dataframe_filtered[['name','categories','distance','lat','lng','id']]
hotels_df.head(10)


# ## 2.e Location of Hotels, Restaurants, Language schools

# In[9]:


hotels_map = folium.Map(location=[latitude, longitude], zoom_start=16) # generate map centred around the Kanyakumari

# add a red circle marker to represent the core location of kanyakumari
folium.features.CircleMarker(
    [latitude, longitude],
    radius=10,
    color='red',
    popup='Pedregalejo beach',
    fill = True,
    fill_color = 'red',
    fill_opacity = 0.6
).add_to(hotels_map)

# add the Italian restaurants as blue circle markers
for lat, lng, label in zip(hotels_df.lat, hotels_df.lng, hotels_df.name):
    folium.features.CircleMarker(
        [lat, lng],
        radius=5,
        color='blue',
        popup=label,
        fill = True,
        fill_color='blue',
        fill_opacity=0.6
    ).add_to(hotels_map)

# display map
hotels_map


# # 3.Methodology section
# 
# 
# In this sections we will perform some data analysis and EDA to find insight from data.We will try to understand the current stats of all given data.Probably,clustering or centroid of all venues will help us to locate new hotel.
# 
# 

# ## 3.a How Far are hotels from the core location

# In[11]:


distance_hotel_df=dataframe_filtered[['name','categories','distance','lat','lng']].sort_values('distance')

def plot_bar_x():
    # this is for plotting purpose
    index = np.arange(len(distance_hotel_df.name))
    plt.bar(distance_hotel_df.name, distance_hotel_df.distance)
    plt.xlabel('Hotels')
    plt.ylabel('Distance from location (Metres)')
    plt.xticks(distance_hotel_df.name,rotation=90)
    plt.title('Langauge Schools, Hotels, Restaurants Vs Distance')
    plt.show()
plot_bar_x()

print("Average distance between Langauge Schools, Hotels, Restaurants and core location is {} metres".format(int(sum(hotels_df['distance'])/hotels_df.shape[0])))


# ## 3.b Explore for other restaurants around Pedregalejo
# 
# A Student always wants to visit new venues.So he wants to reside somewhere nearby to all restaurants.We will be exploring more venues around the core location.We will be digging more on main areas or place around 500 metres.

# In[45]:


radius=400
url_venues = 'https://api.foursquare.com/v2/venues/explore?client_id={}&client_secret={}&ll={},{}&v={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET, latitude, longitude, VERSION, radius, LIMIT)
#url_venues


# In[46]:


import requests

results_venues = requests.get(url_venues).json()
'There are {} restaurants around Pedregalejo.'.format(len(results_venues['response']['groups'][0]['items']))


# In[47]:



items_venues = results_venues['response']['groups'][0]['items']
#items_venues[0]


# In[48]:


dataframe_venues = json_normalize(items_venues) # flatten JSON

# filter columns
filtered_columns = ['venue.name', 'venue.categories'] + [col for col in dataframe_venues.columns if col.startswith('venue.location.')] + ['venue.id']
dataframe_filtered_venues = dataframe_venues.loc[:, filtered_columns]

# filter the category for each row
dataframe_filtered_venues['venue.categories'] = dataframe_filtered_venues.apply(get_category_type, axis=1)

# clean columns
dataframe_filtered_venues.columns = [col.split('.')[-1] for col in dataframe_filtered_venues.columns]

dataframe_filtered_venues.name


# ## 3.c Extract Venues using Search Queries
# 
# Below is the function to extract only Restaurants as wealthy students will be intreseted in Restaurants and bars.

# In[49]:



# Data extracted from foursquare venues
four_sq_venue=pd.DataFrame(dataframe_filtered_venues[['name','categories','distance','lat','lng','id']])

# Data extracted from search queries
new_venues=pd.DataFrame(search_df)

# Concatenate both dataframe
df_venue=pd.concat([four_sq_venue, new_venues],sort=True)


# In[54]:


to_drop = ['Restautant']
df_venues = df_venue[~df_venue['name'].str.contains('|'.join(to_drop))].reset_index()
print("There are {} Restaurant and bars in Pedregalejo".format(df_venues.shape[0]))
df_venues[['name','distance','id']]


# ## 3.d Location of all venues

# In[52]:



venues_map = folium.Map(location=[latitude, longitude], zoom_start=16) # generate map centred around the Conrad Hotel

# add a red circle marker to represent the Kanyakumari
folium.features.CircleMarker(
    [latitude, longitude],
    radius=10,
    color='red',
    popup='Pedregalejo',
    fill = True,
    fill_color = 'red',
    fill_opacity = 0.6
).add_to(venues_map)

# add the Italian restaurants as blue circle markers
for lat, lng, label in zip(df_venues.lat, df_venues.lng, df_venues.name):
    folium.features.CircleMarker(
        [lat, lng],
        radius=5,
        color='black',
        #popup=label,
        fill = True,
        fill_color='black',
        fill_opacity=0.6
    ).add_to(venues_map)

# display map
venues_map


# In[ ]:


All venues seems to be dispersed except seashore areas.

We have listed out number of hotels and venues around kanyakumari.There are 36 Hotels/Restaurant and 17 Venues


# ## 3.e How far are venues from the core location?

# In[55]:


distance_venues_df=df_venues.sort_values('distance')

def plot_bar_venue():
    # this is for plotting purpose
    index = np.arange(len(distance_venues_df.name))
    plt.bar(distance_venues_df.name, distance_venues_df.distance)
    plt.xlabel('Venues')
    plt.ylabel('Distance from location (Metres)')
    plt.xticks(distance_venues_df.name,rotation=90)
    plt.title('Famous Restaurants and bars Vs Distance')
    plt.show()
plot_bar_venue()


# ## 3.F Venue Categories

# In[56]:



freq_venue=df_venues['categories'].value_counts()
freq_venue=pd.DataFrame(freq_venue).reset_index()
freq_venue.columns=['Category','Count']
freq_venue

def plot_bar_categ():
    # this is for plotting purpose
    index = np.arange(len(freq_venue.Category))
    plt.bar(freq_venue.Category, freq_venue.Count)
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.xticks(freq_venue.Category,rotation=90)
    plt.title('Venue Categories')
    plt.show()
plot_bar_categ()


# ## 3.G Rating of all Venues

# In[58]:


#Rating of venues
rating_df=[]

for k in range(df_venues.shape[0]):
    venue_id=df_venues.id[k]
    url = 'https://api.foursquare.com/v2/venues/{}?client_id={}&client_secret={}&v={}'.format(venue_id, CLIENT_ID, CLIENT_SECRET, VERSION)
    result = requests.get(url).json()
    #print(result)
    try:
        #print(df_venues.name[k],result['response']['venue']['rating'])
        rating=result['response']['venue']['rating']
        rating_df.append(rating)
        
    except:
        #print(df_venues.name[k],'This venue has not been rated yet.')
        rating='No Rating Yet'
        rating_df.append(rating)


# In[59]:


rate_dict = {'Venue': df_venues.name, 'Rating': rating_df,'distance':df_venues.distance}
rate_df=pd.DataFrame(rate_dict)
rate_df


# In[62]:


#Lets take values of only rated venues
only_rated_tips = rate_df[(rate_df['Rating']!='No Rating Yet')]

only_rated_tips.reset_index(inplace = True,drop = True) 
only_rated_tips


# ## 3.H Final list of Restaunat and bars

# In[63]:


rated_list=[]
for i in range(len(only_rated_tips)):
    rated_tip_temp=only_rated_tips['Venue'][i]
    rated_list.append(rated_tip_temp)

#Masking all values present in list
mask = df_venues['name'].isin(rated_list)

final_venues = df_venues[mask]
#final_venues['location']=final_venues['lat'].astype(str).str.cat(final_venues['lng'].astype(str), sep=' - ')
final_venues.reset_index(inplace = True,drop = True) 

final_venues


# ## 3.I Clustering based on venues

# In[64]:


# one hot encoding
neighbor_onehot = pd.get_dummies(final_venues[['categories']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
neighbor_onehot['name'] = final_venues['name'] 

# move neighborhood column to the first column
fixed_columns = [neighbor_onehot.columns[-1]] + list(neighbor_onehot.columns[:-1])
neighbor_onehot = neighbor_onehot[fixed_columns]

neighbor_onehot.head()


# In[65]:


neighbor_onehot.shape
neighbor_grouped = neighbor_onehot.groupby('name').mean().reset_index()


# In[66]:


# Top 10 venues
num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['name']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['name'] = neighbor_grouped['name']


# In[67]:


# Clustering

# set number of clusters
kclusters = 3

neighbor_grouped_clustering = neighbor_grouped.drop('name', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(neighbor_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 


# add clustering labels
neighborhoods_venues_sorted.insert(0, 'Clustersss', kmeans.labels_)

neighbor_merged = final_venues

# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
neighbor_merged = neighbor_merged.join(neighborhoods_venues_sorted.set_index('name'), on='name')

kmeans


# ## 3.J Center of all clusters & Midpoint of all venues
# 
# We will be collating the location of centroid of all clusters and midpoint of all venues to get more accurate location

# In[68]:


fin=neighbor_merged.groupby(['Clustersss']).mean()

lati=sum(fin.lat)/len(fin.lat)
longi=sum(fin.lng)/len(fin.lng)

#Taking midpoint of top ten closest hotel
venues_lan=sum(final_venues.lat)/len(final_venues.lat)
venues_lng=sum(final_venues.lng)/len(final_venues.lng)

final_latitude=(lati+venues_lan)/2
final_longitude=(longi+venues_lng)/2

print("Final location (Green Dot in our below given map) of our brand new Student Residence:{},{}".format(final_latitude,final_longitude))


# In[69]:


map_clusters = folium.Map(location=[latitude, longitude], zoom_start=17)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

folium.features.CircleMarker(
    [final_latitude, final_longitude],
    radius=11,
    color='green',
    popup='My hotel',
    fill = True,
    fill_color = 'green',
    fill_opacity = 0.8
).add_to(map_clusters)

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(neighbor_merged['lat'], neighbor_merged['lng'], neighbor_merged['name'], neighbor_merged['Clustersss']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=6,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)

       
map_clusters


# # 4. Results section
# 
# ## 4.a My Student Residence location
# 
# Final location at Student Residence:36.72097593712997,-4.372508839314111
# 
# Surrounded by plenty variety of bars and restaurants. In addition, nearby Language Schools.
# 
# ## 4.b Top Rated Restaurant and bars
# 
# La Galerna	8.4
# Pedregalejo	8.7
# Paseo Marítimo El Pedregal	7.8
# 
# Mafalda	8	
# 
# La Chancla Hotel	7.8
# 
# 
# 
# ## 4.c Spot my Student residence against others
# Green - My Student Residence location
# Red - Pedregalejo core location.
# Black - Restaurant and bars.
# Blue - Other hotels.
# 
# My predicted location and core location are very close to each other which is expected.As this has central attraction,the predicted one almost matched with the core.

# In[74]:


# add a red circle marker to represent the my hotel location
folium.features.CircleMarker(
    [final_latitude, final_longitude],
    radius=20,
    color='green',
    popup='My Hotel',
    fill = True,
    fill_color = 'green',
    fill_opacity = 0.6
).add_to(my_hotel_vs_all)


# add a red circle marker to represent the core location of kanyakumari
folium.features.CircleMarker(
    [latitude, longitude],
    radius=10,
    color='red',
    popup='Pedregalejo',
    fill = True,
    fill_color = 'red',
    fill_opacity = 0.6
).add_to(my_hotel_vs_all)

# add the Italian restaurants as blue circle markers
for lat, lng, label in zip(hotels_df.lat, hotels_df.lng, hotels_df.name):
    folium.features.CircleMarker(
        [lat, lng],
        radius=5,
        color='blue',
        popup=label,
        fill = True,
        fill_color='blue',
        fill_opacity=0.6
    ).add_to(my_hotel_vs_all)
    
    


# display map
my_hotel_vs_all


# ## 4.d Few more Stats
# 
# Hotel La chancla, ,Hotel Elcano will be our competitors as they stay close to our predicted location.
# 
# # 5. Discussion section
# From above reports,we could get an idea why the predicted one is pointed/clustered on the given spot.First most thing could be the center of attraction for the place.
# 
# KMeans have figured out the most common place for all the venues.This output was very adjacent to the core location.This proves the accurate spotting of our predicted algorithm.
# 
# 
# # 6. Conclusion section
# As a business person would be able to set up a student residence on given spot, surrounded by excellent bars and hotels.This will bring revenue automatically as we have located in very near to core one.We proved this with Kmeans.
# 
# My Experience:
# 
# It was wonderful journey for me in IBM capstone and other courses.It can aid to layman people as well who dont know a pinch of Data science.Thanks to Coursera for keeping Skilful instructors with their awesome materials
# 
