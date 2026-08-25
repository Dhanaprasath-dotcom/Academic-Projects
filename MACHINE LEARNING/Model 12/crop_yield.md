```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```


```python
df=pd.read_csv("crop_yield.csv")
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Crop</th>
      <th>Crop_Year</th>
      <th>Season</th>
      <th>State</th>
      <th>Area</th>
      <th>Production</th>
      <th>Annual_Rainfall</th>
      <th>Fertilizer</th>
      <th>Pesticide</th>
      <th>Yield</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Arecanut</td>
      <td>1997</td>
      <td>Whole Year</td>
      <td>Assam</td>
      <td>73814.0</td>
      <td>56708</td>
      <td>2051.4</td>
      <td>7024878.38</td>
      <td>22882.34</td>
      <td>0.796087</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Arhar/Tur</td>
      <td>1997</td>
      <td>Kharif</td>
      <td>Assam</td>
      <td>6637.0</td>
      <td>4685</td>
      <td>2051.4</td>
      <td>631643.29</td>
      <td>2057.47</td>
      <td>0.710435</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Castor seed</td>
      <td>1997</td>
      <td>Kharif</td>
      <td>Assam</td>
      <td>796.0</td>
      <td>22</td>
      <td>2051.4</td>
      <td>75755.32</td>
      <td>246.76</td>
      <td>0.238333</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Coconut</td>
      <td>1997</td>
      <td>Whole Year</td>
      <td>Assam</td>
      <td>19656.0</td>
      <td>126905000</td>
      <td>2051.4</td>
      <td>1870661.52</td>
      <td>6093.36</td>
      <td>5238.051739</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Cotton(lint)</td>
      <td>1997</td>
      <td>Kharif</td>
      <td>Assam</td>
      <td>1739.0</td>
      <td>794</td>
      <td>2051.4</td>
      <td>165500.63</td>
      <td>539.09</td>
      <td>0.420909</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>19684</th>
      <td>Small millets</td>
      <td>1998</td>
      <td>Kharif</td>
      <td>Nagaland</td>
      <td>4000.0</td>
      <td>2000</td>
      <td>1498.0</td>
      <td>395200.00</td>
      <td>1160.00</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>19685</th>
      <td>Wheat</td>
      <td>1998</td>
      <td>Rabi</td>
      <td>Nagaland</td>
      <td>1000.0</td>
      <td>3000</td>
      <td>1498.0</td>
      <td>98800.00</td>
      <td>290.00</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>19686</th>
      <td>Maize</td>
      <td>1997</td>
      <td>Kharif</td>
      <td>Jammu and Kashmir</td>
      <td>310883.0</td>
      <td>440900</td>
      <td>1356.2</td>
      <td>29586735.11</td>
      <td>96373.73</td>
      <td>1.285000</td>
    </tr>
    <tr>
      <th>19687</th>
      <td>Rice</td>
      <td>1997</td>
      <td>Kharif</td>
      <td>Jammu and Kashmir</td>
      <td>275746.0</td>
      <td>5488</td>
      <td>1356.2</td>
      <td>26242746.82</td>
      <td>85481.26</td>
      <td>0.016667</td>
    </tr>
    <tr>
      <th>19688</th>
      <td>Wheat</td>
      <td>1997</td>
      <td>Rabi</td>
      <td>Jammu and Kashmir</td>
      <td>239344.0</td>
      <td>392160</td>
      <td>1356.2</td>
      <td>22778368.48</td>
      <td>74196.64</td>
      <td>1.261818</td>
    </tr>
  </tbody>
</table>
<p>19689 rows × 10 columns</p>
</div>




```python
df.isnull().sum()
```




    Crop               0
    Crop_Year          0
    Season             0
    State              0
    Area               0
    Production         0
    Annual_Rainfall    0
    Fertilizer         0
    Pesticide          0
    Yield              0
    dtype: int64




```python
print("Logistic regression")
print('-'*30)
```

    Logistic regression
    ------------------------------
    


```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,precision_score,recall_score
```


```python

```
