
# coding: utf-8

# In[2]:


import urllib.request
import json
import pandas as pd
from pandas.io.json import json_normalize


# In[3]:


def getRequestUrl(url):
    response=urllib.request.urlopen(url)
    return response.read().decode("utf-8")


# In[4]:


def getMsgData(pageNum):
    url='http://apis.data.go.kr/1741000/DisasterMsg2/getDisasterMsgList?ServiceKey=#########&type=json&pageNo='
    url+=str(pageNum)
    json_str=getRequestUrl(url)
    json_object=json.loads(json_str)
    return json_normalize(json_object['DisasterMsg'][1]['row'])


# In[5]:


import numpy as np
from pandas import DataFrame


# In[6]:


def ItemList(d_list):
    for line in d_list['create_date']:
        all_dates.append(line[:10])


# In[22]:


all_dates=[]
for i in range(1,1172):
    ItemList(getMsgData(i))
all_dates.reverse()


# In[23]:


items=pd.unique(all_dates)
zero=np.zeros(((1, len(items))))
dummy=DataFrame(zero,columns=items)


# In[24]:


for n in all_dates:
    for m in items:
        if n==m:
            dummy.ix[0,n]+=1  
items_matrix=dummy.T.sum(axis=1)


# In[25]:


import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
get_ipython().run_line_magic('matplotlib', 'inline')


# In[26]:


path_font='C:/Users/MHY/.fontconfig/NanumGothic.ttf'
fontprop=fm.FontProperties(fname=path_font,size=15)
plt.style.use('ggplot')


# In[27]:


items_matrix.plot(kind='bar')
plt.title('일별 재난문자 발생 수', fontsize=20, fontproperties=fontprop)
plt.xlabel('일', fontsize=20, fontproperties=fontprop)
plt.ylabel('재난문자 수', fontsize=20, fontproperties=fontprop)
plt.savefig('C:/Users/MHY/Desktop/논문/chart1.png', bbox_inches='tight', dpi=300)
plt.show()


# In[133]:


items_matrix.plot(kind='line')
plt.title('일별 재난문자 발생 수', fontsize=20, fontproperties=fontprop)
#plt.xticks(np.arange(10))
plt.xlabel('일', fontsize=20, fontproperties=fontprop)
plt.ylabel('재난문자 수', fontsize=20, fontproperties=fontprop)
plt.savefig('C:/Users/MHY/Desktop/논문/chart2.png', bbox_inches='tight', dpi=300)
plt.show()


# In[96]:


items_matrix


# In[29]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(items_matrix)
plot_pacf(items_matrix)
plt.show()


# In[32]:


diff_1=items_matrix.diff(periods=1).iloc[1:]
diff_1.plot()
plot_acf(diff_1)
plot_pacf(diff_1)
plt.show()


# In[126]:


from statsmodels.tsa.arima_model import ARIMA
model=ARIMA(items_matrix, order=(1,0,2))
model_fit=model.fit(trend='nc', full_output=True, disp=1)
print(model_fit.summary())


# In[127]:


model_fit.plot_predict()
plt.savefig('C:/Users/MHY/Desktop/논문/(1,0,2).png', bbox_inches='tight', dpi=300)


# In[128]:


fore=model_fit.forecast(steps=1)
print(fore)


# In[132]:


items[14]

