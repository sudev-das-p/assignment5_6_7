#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model


# # Assignment5
# 

# ## Ground Cricket Chirps
# 
# In _The Song of Insects_ (1948) by George W. Pierce, Pierce mechanically measured the frequency (the number of wing vibrations per second) of chirps (or pulses of sound) made by a striped ground cricket, at various ground temperatures.  Since crickets are ectotherms (cold-blooded), the rate of their physiological processes and their overall metabolism are influenced by temperature.  Consequently, there is reason to believe that temperature would have a profound effect on aspects of their behavior, such as chirp frequency.
# 
# In general, it was found that crickets did not sing at temperatures colder than 60º F. or warmer than 100º F.

# In[3]:


ground_cricket_data = {"Chirps/Second": [20.0, 16.0, 19.8, 18.4, 17.1, 15.5, 14.7,
                                         15.7, 15.4, 16.3, 15.0, 17.2, 16.0, 17.0,
                                         14.4],
                       "Ground Temperature": [88.6, 71.6, 93.3, 84.3, 80.6, 75.2, 69.7,
                                              71.6, 69.4, 83.3, 79.6, 82.6, 80.6, 83.5,
                                              76.3]}
df = pd.DataFrame(ground_cricket_data)


# In[4]:


X = df.iloc[:,:-1]
y = df.iloc[:,-1]


# In[43]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X,y)
slope = model.coef_[0]
intercept = model.intercept_
print("Linear regression equation for the given data is y =",format(round(slope,2)),"X + ",format(round(intercept,2)))
plt.scatter(X,y)
plt.plot(X,slope*X+intercept)
plt.show()
print(model.score(X,y))
temp = model.predict([[18]])[0]
print("predicted temperature for chirp rate of 18 chirps/sec",temp)
chirp = (95-intercept)/slope
print("predicted chirp rate for temperature of 95 is",chirp)


# ### Tasks
# 
# 1. Find the linear regression equation for this data.
# 2. Chart the original data and the equation on the chart.
# 3. Find the equation's $R^2$ score (use the `.score` method) to determine whether the
# equation is a good fit for this data. (0.8 and greater is considered a strong correlation.)
# 4. Extrapolate data:  If the ground temperature reached 95, then at what approximate rate would you expect the crickets to be chirping?
# 5. Interpolate data:  With a listening device, you discovered that on a particular morning the crickets were chirping at a rate of 18 chirps per second.  What was the approximate ground temperature that morning? 

# # Assignment6

# ## Brain vs. Body Weight
# 
# In the file `brain_body.txt`, the average brain and body weight for a number of mammal species are recorded. Load this data into a Pandas data frame.
# 
# ### Tasks
# 
# 1. Find the linear regression equation for this data for brain weight to body weight.
# 2. Chart the original data and the equation on the chart.
# 3. Find the equation's $R^2$ score (use the `.score` method) to determine whether the
# equation is a good fit for this data. (0.8 and greater is considered a strong correlation.)

# In[68]:


df = pd.read_fwf("brain_body.txt")
x = df.iloc[:,:-1]
y = df.iloc[:,-1]
model = LinearRegression()
model.fit(x,y)
slope = model.coef_[0]
intercept = model.intercept_
print("Linear regression equation for the given data is y =",format(round(slope,2)),"X + ",format(round(intercept,2)))
plt.scatter(x,y)
plt.plot(x,slope*x+intercept,color = 'y')
print(model.score(x,y))


# # Assignment7

# ## Salary Discrimination
# 
# The file `salary.txt` contains data for 52 tenure-track professors at a small Midwestern college. This data was used in legal proceedings in the 1980s about discrimination against women in salary.
# 
# The data in the file, by column:
# 
# 1. Sex. 1 for female, 0 for male.
# 2. Rank. 1 for assistant professor, 2 for associate professor, 3 for full professor.
# 3. Year. Number of years in current rank.
# 4. Degree. Highest degree. 1 for doctorate, 0 for master's.
# 5. YSdeg. Years since highest degree was earned.
# 6. Salary. Salary/year in dollars.
# 
# ### Tasks
# 
# 1. Find the linear regression equation for this data using columns 1-5 to column 6.
# 2. Find the selection of columns with the best $R^2$ score.
# 3. Report whether sex is a factor in salary.

# In[3]:


df = pd.read_fwf("salary.txt", header=None, 
                 names=["Sex", "Rank", "Year", "Degree", "YSdeg", "Salary"])


# In[27]:


from sklearn.linear_model import LinearRegression
x = df.iloc[:,:-1]
y = df.iloc[:,-1]
model = LinearRegression()
model.fit(x,y)
slope = model.coef_
c1 = slope[0]
c2 = slope[1]
c3 = slope[2]
c4 = slope[3]
c5 = slope[4]
intercept = model.intercept_
print("Linear regression equation for the given data is y =",format(round(c1,2)),"X1 + ",format(round(c2,2)),"X2 + ",format(round(c3,2)),"X3 ",format(round(c4,2)),"X4 ",format(round(c5,2)),"X5 ")


# In[29]:


from itertools import combinations
def linearModels(features):
    lr = LinearRegression()
    lr.fit(x[list(features)],y)
    return lr.score(x[list(features)],y)
best_score = 0
best_features = []
for i in range(1,len(x.columns)+1):
    for combo in combinations(x.columns,i):
        score = linearModels(combo)
        if score>best_score:
            best_score = score
            best_combo = combo
print("best feature combo : ",best_combo," , best r2 score : ",best_score)


# In[4]:


import seaborn as sns
sns.heatmap(df.corr())


# In[17]:


plt.scatter(df['Rank'],df['Salary'],c = df['Sex'])
print(df[(df['Sex']==1) & (df['Rank']==3)]['Salary'].count())
print(df[(df['Sex']==0) & (df['Rank']==3)]['Salary'].count())
df[df['Sex']==0].count()

