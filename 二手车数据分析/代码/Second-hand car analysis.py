#!/usr/bin/env python
# coding: utf-8

# # Second-hand car analysis

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
df = pd.read_csv("autos.csv",encoding = 'cp1252')


# ## Brief view

# In[2]:


df.head()


# In[3]:


df.columns


# In[4]:


c = ('seller','nrOfPictures','offerType','abtest','vehicleType','brand','yearOfRegistration')


# In[5]:


for i in c:
    print(df.groupby(i).count()['name'],'\n')


# In[6]:


print("Too new: %d" % df.loc[df.yearOfRegistration >= 2017].count()['name'])
print("Too old: %d" % df.loc[df.yearOfRegistration < 1950].count()['name'])
print("Too cheap: %d" % df.loc[df.price < 100].count()['name'])
print("Too expensive: " , df.loc[df.price > 150000].count()['name'])
print("Too few km: " , df.loc[df.kilometer < 5000].count()['name'])
print("Too many km: " , df.loc[df.kilometer > 200000].count()['name'])
print("Too few PS: " , df.loc[df.powerPS < 10].count()['name'])
print("Too many PS: " , df.loc[df.powerPS > 500].count()['name'])


# ## Discard useless data

# In[7]:


df = df.drop_duplicates(['name','price','vehicleType','yearOfRegistration'
                         ,'gearbox','powerPS','model','kilometer','monthOfRegistration','fuelType'
                         ,'notRepairedDamage'])


# In[8]:


df.drop(['seller','offerType','postalCode','dateCreated','lastSeen','dateCrawled','nrOfPictures','abtest']
        ,axis = 1, inplace = True)


# In[9]:


df.describe()


# In[10]:


df= df[(df['price']<=200000) & (df['price']>100) 
       & (df['yearOfRegistration']>1960) & (df['yearOfRegistration']<=2016)
      &(df['powerPS'] <= 1000) & (df['powerPS'] >= 10)] 


# In[11]:


df.describe()


# ## Add useful columns

# In[12]:


df['car_age'] = (2016) - (df['yearOfRegistration'])


# In[13]:


df['horse_power'] = (df['powerPS'] * 0.986)


# In[14]:


df.describe()


# In[15]:


df.head()


# ## Translate data into english & create mandarin dictionary

# In[16]:


trans_eng = {'automatik':'automatic','manuell':'manual','kombi':'combi','andere':'others'
             ,'kleinwagen':'supermini','cabrio':'convertible','benzin':'gasoline'
             ,'elektro':'electro','nein':'no','ja':'yes','sonstige_autos':'miscellaneous_car'}
for i in trans_eng:
    df.replace(i,trans_eng[i],inplace=True)


# In[17]:


trans_man = {'name':'Name','price':'价格','vehicleType':'种类','monthOfRegistration':'注册月份','powerPS':'公制马力'
             ,'yearOfRegistration':'注册年份','gearbox':'变速箱','horse_power':'马力',
             'model':'型号','kilometer':'里程','fuelType':'燃料','brand':'品牌','notRepairedDamage':'受损情况','car_age':'车龄',
            'not-declared':'未知','coupe':'双座四轮轿车','suv':'运动型多功能车','supermini':'迷你车',
             'limousine':'加长轿车','convertible':'敞篷车','bus':'巴士','combi':'旅行车',
             'others':'其他','gasoline':'汽油','diesel':'柴油','nan':'未知','lpg':'液化石油气',
             'hybrid':'油电混合','cng':'压缩天然气', 'electro':'电动','manual':'手动档','automatic':'自动档',
             'volkswagen':'大众汽车','audi':'奥迪','jeep':'吉普车','skoda':'斯柯达','bmw':'宝马','peugeot':'标致',
             'ford':'福特','mazda':'长安马自达','nissan':'日产','renault':'雷诺','mercedes_benz':'奔驰',
             'seat':'西雅特','honda':'本田','fiat':'菲亚特','opel':'欧宝','mini':'宝马迷你','smart':'精灵汽车',
             'hyundai':'现代汽车','alfa_romeo':'阿尔法·罗密欧','subaru':'斯巴鲁','volvo':'沃尔沃','mitsubishi':'三菱',
             'kia':'起亚','suzuki':'铃木','lancia':'蓝旗亚','porsche':'保时捷','citroen':'雪铁龙',
             'toyota':'丰田','chevrolet':'雪佛兰','dacia':'达西亚','daihatsu':'大发汽车','trabant':'卫星轿车',
             'chrysler':'克莱斯勒','jaguar':'捷豹汽车','daewoo':'大宇汽车','rover':'罗孚','saab':'萨博','land_rover':'路虎',
             'lada':'拉达汽车','no':'无','yes':'有','golf':'高尔夫','miscellaneous_car':'其他'}


# In[18]:


df.head()


# ## Deal with null data & simple visualization

# In[19]:


df.isnull().sum()


# ### Fill with  'not-declared'

# In[20]:


na_set = ['vehicleType','gearbox','model','fuelType','notRepairedDamage']
df_isna = df.copy()
for i in na_set:
    df_isna[i].fillna('not-declared',inplace = True)


# In[21]:


categories = ['gearbox','notRepairedDamage','fuelType','model', 'brand', 'vehicleType']
for i, c in enumerate(categories):
    g = df_isna.groupby(by=c)[c].count().sort_values(ascending=False)
    labels = g.index
    if(c != 'model'):
        translated  = []    
        for i in labels:
            translated.append(trans_man[i])
        labels = translated
    l = len(labels)
    if(l > 5):
        print(g.head())
        r = range(min(l,5))
        plt.figure(figsize=(5,3))
        plt.bar(r, g.head(),color = '#aabbcc')
        plt.xticks(r, labels)
        plt.ylabel("数量")
        plt.title(trans_man[c],fontdict = {'fontweight':'bold','fontsize':18})
        plt.show()
        
    else:
        print(g)
        plt.figure(figsize=(5,3))
        colors = ['#abcdef','#aabbcc','#C2CFDC']
        plt.pie(g,labels=labels,autopct='%1.1f%%',shadow=False, startangle=90,colors = colors)
        plt.title(trans_man[c], fontdict = {'fontweight':'bold','fontsize':13})
        plt.show()


# ### Back fill & Front fill

# In[22]:


df_fillna = df.copy()
for i in na_set:
    df_fillna[i].fillna(method='ffill',inplace=True)
for i in na_set:
    df_fillna[i].fillna(method='bfill',inplace=True)


# In[23]:


for i, c in enumerate(categories):
    g = df_fillna.groupby(by=c)[c].count().sort_values(ascending=False)
    labels = g.index
    if(c != 'model'):
        translated  = []    
        for i in labels:
            translated.append(trans_man[i])
        labels = translated
    l = len(labels)
    if(l > 5):
        print(g.head())
        r = range(min(l,5))
        plt.figure(figsize=(5,3))
        plt.bar(r, g.head(),color = '#aabbcc')
        plt.xticks(r, labels)
        plt.ylabel("数量")
        plt.title(trans_man[c],fontdict = {'fontweight':'bold','fontsize':18})
        plt.show()
        
    else:
        print(g)
        plt.figure(figsize=(5,3))
        colors = ['#abcdef','#aabbcc','#C2CFDC']
        plt.pie(g,labels=labels,autopct='%1.1f%%',shadow=False, startangle=90,colors = colors)
        plt.title(trans_man[c], fontdict = {'fontweight':'bold','fontsize':13})
        plt.show()


# ### Drop all na

# In[24]:


df_dropna = df.copy().dropna()


# In[25]:


for i, c in enumerate(categories):
    g = df_dropna.groupby(by=c)[c].count().sort_values(ascending=False)
    labels = g.index
    if(c != 'model'):
        translated  = []    
        for i in labels:
            translated.append(trans_man[i])
        labels = translated
    l = len(labels)
    if(l > 5):
        print(g.head())
        r = range(min(l,5))
        plt.figure(figsize=(5,3))
        plt.bar(r, g.head(),color = '#aabbcc')
        plt.xticks(r, labels)
        plt.ylabel("数量")
        plt.title(trans_man[c],fontdict = {'fontweight':'bold','fontsize':18})
        plt.show()
        
    else:
        print(g)
        plt.figure(figsize=(5,3))
        colors = ['#abcdef','#aabbcc','#C2CFDC']
        plt.pie(g,labels=labels,autopct='%1.1f%%',shadow=False, startangle=90,colors = colors)
        plt.title(trans_man[c], fontdict = {'fontweight':'bold','fontsize':13})
        plt.show()


# In[26]:


df = df_fillna.copy()


# ## Correlation analysis and more visualization

# In[27]:


plt.figure(figsize=(5,3))
r = range(5)
prices = df.groupby(['brand']).mean()['price'].sort_values(ascending=False)
plt.bar(r,prices.head(),color = '#aabbcc')
labels = prices.index
translated  = []    
for i in labels:
    translated.append(trans_man[i])
labels = translated
plt.xticks(r,labels,size = 10)
plt.ylabel('Price')
plt.title('价格与品牌的关系',fontdict = {'fontweight':'bold','fontsize':18})
plt.show()


# In[64]:


km = df.groupby(['kilometer'])
num = km.count()['name']

km = [km for km, df in km]

plt.scatter(km, num, color = '#e07b39')
plt.xticks(km, rotation = 'vertical', size = 8)
plt.xlabel('公里数')
plt.ylabel('汽车的数量')
plt.title('公里数与汽车数量的关系',fontdict = {'fontweight':'bold','fontsize':18})
plt.show()


# In[65]:


plt.figure(figsize = (15,5))
prices = df.groupby(['kilometer']).mean()['price']
plt.plot(km,prices)
plt.grid()
plt.xticks(km, rotation = 'vertical', size = 10)
plt.xlabel('公里数')
plt.ylabel('价钱')
plt.title('公里数与价钱的关系',fontdict = {'fontweight':'bold','fontsize':18})
plt.show()


# In[30]:


from matplotlib import cm
color = cm.inferno_r(np.linspace(.4, .8, 30))
##color


# In[66]:


## Method 1
## Show out the relationship respectively in two graphs

## Find out the relationship between brand and number of cars sold
plt.figure(figsize = (15,5))
brand = df.groupby('brand')
num = brand.count()['car_age']

brands = [brand for brand, df in brand]
plt.grid()

plt.bar(brands,num, color = color)
plt.xticks(brands, rotation = 'vertical', size = 8)
plt.xlabel('品牌')
plt.ylabel('汽车的数量')
plt.title('品牌与汽车的数量的关系',fontdict = {'fontweight':'bold','fontsize':18})
plt.show()

## Volkswagen is the most, follow by bmw and opel


# In[68]:


## Find out which brand sold with the highest price
plt.figure(figsize = (15,5))
prices = df.groupby(['brand']).mean()['price']
plt.plot(brands,prices,color = '#fc035e')
plt.grid()

plt.xticks(brands, rotation = 'vertical', size = 10)
plt.xlabel('品牌')
plt.ylabel('价钱')
plt.title('品牌与价钱的关系',fontdict = {'fontweight':'bold','fontsize':18})
plt.show()
## Porsche sold with a more highest price, follow by land_rover and sonstige_autos


# In[69]:


car_age = [car_age for car_age, d in df.groupby('car_age')]
price = df.groupby('car_age').mean()['price']

fig,ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.bar(car_age,price,color = 'y')
ax2.plot(car_age, df.groupby(['car_age']).count(),'b')
ax1.grid()
ax2.set_title('车龄与其数量和价格的关系',fontdict = {'fontweight':'bold','fontsize':18})
ax2.set_xlabel('车龄')
ax2.set_ylabel('数量',color = 'b')
ax1.set_ylabel('价格',color = 'y')
plt.show()


# In[34]:


df_corr = df.corr(method = 'pearson')
df_corr.columns
labels = [trans_man[c] for c in df_corr.columns]
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
ax.matshow(df_corr,cmap=plt.cm.RdYlGn)

ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
plt.show()


# In[35]:


df_corr = df.corr(method = 'kendall')
df_corr.columns
labels = [trans_man[c] for c in df_corr.columns]
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
ax.matshow(df_corr,cmap=plt.cm.RdYlGn)

ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
plt.show()


# In[36]:


df_corr = df.corr(method = 'spearman')
df_corr.columns
labels = [trans_man[c] for c in df_corr.columns]
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
ax.matshow(df_corr,cmap=plt.cm.RdYlGn)

ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
plt.show()


# In[37]:


#### Functions
def category_values(dataframe, categories):
    for c in categories:
        print('\n', dataframe.groupby(by=c)[c].count().sort_values(ascending=False))
        print('Nulls: ', dataframe[c].isnull().sum())

def plot_correlation_map( df ):
    corr = df.corr(method = 'pearson')
    _ , ax = plt.subplots( figsize =( 12 , 10 ) )
    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
    _ = sns.heatmap(
        corr, 
        cmap = cmap,
        square=True, 
        cbar_kws={ 'shrink' : .9 }, 
        ax=ax, 
        annot = True, 
        annot_kws = { 'fontsize' : 12 }
    )


# In[38]:


from sklearn import datasets, linear_model, preprocessing, svm
from sklearn.preprocessing import StandardScaler, Normalizer

labels = ['name', 'gearbox', 'notRepairedDamage', 'model', 'brand', 'fuelType', 'vehicleType']
les = {}

df2 = df.copy()
for l in labels:
    les[l] = preprocessing.LabelEncoder()
    les[l].fit(df2[l])
    tr = les[l].transform(df2[l]) 
    df2.loc[:, l + '_feat'] = pd.Series(tr, index=df2.index)

labeled = df2[ ['price','horse_power','kilometer','car_age']+ [x+"_feat" for x in labels]].copy()
len(labeled['name_feat'].unique()) / len(labeled['name_feat'])
labeled.drop(['name_feat'], axis='columns', inplace=True)
#print (labeled)
plot_correlation_map(labeled)
labeled.corr()


# In[39]:


from sklearn import datasets, linear_model, preprocessing, svm
from sklearn.preprocessing import StandardScaler, Normalizer
df2 = df.copy()
for i in trans_man:
    df2.replace({i:trans_man[i]},inplace=True)


brand_counts=df2['brand'].value_counts(normalize=True)
common_brands=brand_counts[brand_counts>0.03].index

#selecting common brands
brand_avg_prices = {}#dictionary

for brand in common_brands:
    brand_only = df2[df2['brand'] == brand]
    avg_price = brand_only['price'].mean()
    brand_avg_prices[brand] = int(avg_price)
    
bap_series = pd.Series(brand_avg_prices).sort_values(ascending = False)
print(pd.DataFrame(bap_series, columns = ["avg_price"]) )
#simply use tables to show

type_counts=df2['vehicleType'].value_counts(normalize=True)
common_types=type_counts[type_counts>0.05].index
#selecting common vehicle types

tri=pd.DataFrame()
for b in list(common_brands):
    for v in list(common_types):
        z=df2[(df2['brand']==b)&(df2['vehicleType']==v)]['price'].mean()
        tri=tri.append(pd.DataFrame({'品牌':b,'种类':v,'平均价格':z},index=[0]))
tri=tri.reset_index()
del tri['index']
tri["平均价格"] = tri["平均价格"].astype(int)

tri = tri.pivot("品牌","种类", "平均价格")
fig, ax = plt.subplots()
sns.heatmap(tri,annot=True,cmap="YlGnBu",fmt="d")
ax.set_title("平均价格和品牌及种类的关系",fontdict={'size':10})
plt.show()


# ## Regression model and price prediction

# In[40]:


df_sci = df.copy()


# In[41]:


c = ['vehicleType','gearbox','model','fuelType','notRepairedDamage','brand','name']
for i in c:
    df_sci[i] = df_sci[i].astype('category').cat.codes


# In[42]:


import sklearn
from sklearn import svm


# In[43]:


df_sci = sklearn.utils.shuffle(df_sci)


# In[44]:


X = df_sci.drop(columns = ['price','name'],axis = 1).values
X = preprocessing.scale(X)
y = df_sci['price'].values


# In[45]:


test_size = 200

X_train = X[:-test_size]
y_train = y[:-test_size]

X_test = X[-test_size:]
y_test = y[-test_size:]

clf = sklearn.svm.LinearSVR()
clf.fit(X_train,y_train)


# In[46]:


clf.score(X_test,y_test)


# In[47]:


for X,y in zip(X_test,y_test):
    print(f"Model:{clf.predict([X])[0]},Actual:{y}")


# In[48]:


clf = sklearn.linear_model.SGDRegressor()
clf.fit(X_train,y_train)


# In[49]:


clf.score(X_test,y_test)


# In[50]:


for X,y in zip(X_test,y_test):
    print(f"Model:{clf.predict([X])[0]},Actual:{y}")


# In[51]:


from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(X_train,y_train)
lr_clf.score(X_test,y_test)


# In[52]:


for X,y in zip(X_test,y_test):
    print(f"Model:{lr_clf.predict([X])[0]},Actual:{y}")


# In[53]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

rf = RandomForestRegressor()

param_grid = { "criterion" : ["mse"]
              , "min_samples_leaf" : [3]
              , "min_samples_split" : [3]
              , "max_depth": [10]
              , "n_estimators": [500]}

gs = GridSearchCV(estimator=rf, param_grid=param_grid, cv=2, n_jobs=-1, verbose=1)
gs = gs.fit(X_train, y_train)


# In[54]:


print(gs.best_score_)
print(gs.best_params_)


# In[55]:


bp = gs.best_params_
forest = RandomForestRegressor(criterion=bp['criterion'],
                              min_samples_leaf=bp['min_samples_leaf'],
                              min_samples_split=bp['min_samples_split'],
                              max_depth=bp['max_depth'],
                              n_estimators=bp['n_estimators'])
forest.fit(X_train, y_train)
# Explained variance score: 1 is perfect prediction
print('Score: %.2f' % forest.score(X_test, y_test))


# In[56]:


for X,y in zip(X_test,y_test):
    print(f"Model:{forest.predict([X])[0]},Actual:{y}")


# In[ ]:




