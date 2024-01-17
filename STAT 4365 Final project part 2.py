#!/usr/bin/env python
# coding: utf-8

# ## A)

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[29]:


df = pd.read_csv('fifa_historic.csv')
df= pd.read_csv('fifa_historic.csv', skiprows=[0, 1, 2])

column_names=['Year', 'Country', 'City', 'Stage', 'Home_Team', 'Away_Team', 'Home_Score', 'Away_Score',
                'Outcome', 'Win_Conditions', 'Winning_Team', 'Losing_Team', 'Date', 'Month', 'DayOfWeek']

df.columns =column_names

#df = df.dropna()
df['Date'] =pd.to_datetime(df['Date'], format='%m/%d/%Y')
df

#I decided to keep the na values because we lose a signifcant amount of information and the graphs become inaccurate representations of the data.


# In[27]:


import matplotlib.pyplot as plt

average_goals_per_year = df.groupby('Year')['Home_Score', 'Away_Score'].mean().mean(axis=1)
plt.figure(figsize=(12, 6))
plt.plot(average_goals_per_year, marker='o', linestyle='-', color='b')
plt.title('Average Goals per Match Over the Years')
plt.xlabel('Year')
plt.ylabel('Average Goals per Match')
plt.grid(True)
plt.show()


df['Score Difference']= df['Home_Score'] - df['Away_Score']
average_score_diff_by_stage = df.groupby('Stage')['Score Difference'].mean()
plt.figure(figsize=(12, 6))
plt.bar(average_score_diff_by_stage.index, average_score_diff_by_stage.values, color='g')
plt.title('Average Score Difference in Various Stages of the Tournament')
plt.xlabel('Tournament Stage')
plt.ylabel('Average Score Difference')
plt.xticks(rotation=45)
plt.show()

average_goals_by_stage =  df.groupby('Stage')['Home_Score', 'Away_Score'].mean().mean(axis=1)
plt.figure(figsize=(12, 6))
plt.bar(average_goals_by_stage.index,average_goals_by_stage.values, color='r')
plt.title('Average Number of Goals in Various Stages of the Tournament')
plt.xlabel('Tournament Stage')
plt.ylabel('Average Number of Goals')
plt.xticks(rotation=45)
plt.show()


# According to the first graph, the average number of goals per match has declined over the years, from around 2-3 goals per match in the 1940s to 1960s to less than 1.5 after the 1960s. The second graph shows the average score difference between teams in various stages of the team. For the most part, the score difference is smaller as the tournament progresses, indicating that the teams are more evenly matched. The third graph shows that the average number of goals scored decreases as the tournament progresses. This also shows that the teams are more evenly matched because they are less likely to score a goal on each other.

# ## B)

# In[26]:


world_cup_finals = df[df['Stage'] == 'Final']
semi_finals = df[df['Stage'] == 'Semifinals']
third_place_matches = df[df['Stage'] == 'Third place']

world_cup_winners = world_cup_finals['Winning_Team'].value_counts()
semi_final_winners = semi_finals['Winning_Team'].value_counts()
third_place_winners = third_place_matches['Winning_Team'].value_counts()

plt.figure(figsize=(15,6))

plt.subplot(1,3,1)
world_cup_winners.plot(kind='bar')
plt.title('World Cup Winners')
plt.xlabel('Team')
plt.ylabel('Number of Wins' )

plt.subplot(1, 3,2)
semi_final_winners.plot(kind='bar')
plt.title('Semi-final Winners')
plt.xlabel('Team')
plt.ylabel('Number of Wins')

plt.subplot(1,  3, 3)
third_place_winners.plot(kind='bar')
plt.title('Third-Place Winners')
plt.xlabel('Team')
plt.ylabel('Number of Wins ')

plt.tight_layout()
plt.show()


# According to the bar graphs above, it looks like Brazil won the most World Cup matches with a total of 5 wins. Italy and Brazil tied for most semi-final wins with a total of 6 wins. West Germany, Brazil, France, Poland, Germany all tied for most third-place wins for a total of 2.

# ## C)

# In[7]:


from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

players_df = pd.read_csv('All_Players .csv')
players_df


# In[8]:


predictors_1 = ['Overall Score', 'Potential Score', 'Height', 'Weight', 'Age']
predictors_2 = ['Ball Skills', 'Defence', 'Mental', 'Passing', 'Physical']
predictors_3 = ['Shooting', 'Goalkeeping']

target = 'Market Value'

cv = KFold(n_splits=10, random_state=1, shuffle=True)

model_1 =LinearRegression()
model_2 =LinearRegression()
model_3 = LinearRegression()

X_model_1 =  players_df[predictors_1]
X_model_2= players_df[predictors_2]
X_model_3 =players_df[predictors_3]
y = players_df[target]


scores_model_1 = cross_val_score(model_1, X_model_1, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
scores_model_2= cross_val_score(model_2, X_model_2, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
scores_model_3 =  cross_val_score(model_3, X_model_3, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)

mae_model_1= np.mean(np.absolute(scores_model_1))
mae_model_2 = np.mean(np.absolute(scores_model_2))
mae_model_3 = np.mean(np.absolute(scores_model_3))
print(f"mae model 1: {mae_model_1}")
print(f"mae model 2: {mae_model_2}")
print(f"mae model 3: {mae_model_3}")


# I split the features in 3 different ways. The first model focussed on physical characteristics of the players, and the players' basic information. The second model focusses on individual skills of the player. The third model only includes shooting and goalkeeping because I feel like those are skills that are important to detemrining a player's overall ability to play on the field. Model 1 has the lowest mean absolute error out of the 3 models, indicating that it is the best at predicting market value of a player. Therefore, a player's basic information is the best indicator of market value. 

# ## D)

# In[9]:


X_model_1_all = players_df[predictors_1]
y_all = players_df[target]

model_1.fit(X_model_1_all, y_all)
mbappe_data= pd.DataFrame([[89, 95, 178, 73, 21]], columns=predictors_1)

mbappe_market_value_prediction =model_1.predict(mbappe_data)

print(f"Predicted Market Value for Mbappe: {mbappe_market_value_prediction[0]}")


# ## E)

# In[10]:


def predict_market_value(overall_score, potential_score, height, weight, age):
    X_model_1_all = players_df[predictors_1]
    y_all = players_df[target]

    model_1.fit(X_model_1_all, y_all)
    player_data = pd.DataFrame([[overall_score, potential_score, height, weight, age]], columns=predictors_1)

    market_value_prediction=model_1.predict(player_data)

    return market_value_prediction

print(predict_market_value(89, 95, 178, 73, 21))

