```python
1a
import pandas as pd import numpy as np
df = pd.read_csv('your_dataset.csv')
print(df.head())
print(df.describe())
print(df.info())
df.dropna(inplace=True)
df.fillna(0, inplace=True)
df.fillna(df.mean(), inplace=True)
df.drop_duplicates(inplace=True)
df['column_name'] = df['column_name'].astype(int)
df['date_column'] = pd.to_datetime(df['date_column'])

2a
import pandas as pd
 df = pd.read_csv('your_data.csv') 
print(df.head()) print(df.describe()) print(df.info())
import matplotlib.pyplot as plt import seaborn as sns
sns.boxplot(data=df)
plt.show()
sns.scatterplot(x='feature1', y='feature2', data=df)
plt.show()
df.hist(bins=50, figsize=(20, 15)) plt.show()
from scipy import stats 
z_scores=stats.zscore(df.select_dtypes(include=[np.number])) abs_z_scores = np.abs(z_scores)
threshold = 3 outliers = (abs_z_scores > threshold).all(axis=1)
outlier_indices = np.where(outliers)[0] print("Outliers detected:",
outlier_indices)
Q1 = df.quantile(0.25) Q3 = df.quantile(0.75) IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR upper_bound = Q3 + 1.5 * IQR
outliers = ((df < lower_bound) | (df > upper_bound)).any(axis=1)
outlier_indices = df.index[outliers] print("Outliers detected:", outlier_indices)
df_cleaned = df.drop(outlier_indices) print(df_cleaned.describe())
sns.boxplot(data=df_cleaned) plt.show() sns.scatterplot(x='feature1',
y='feature2', data=df_cleaned) plt.show() df_cleaned.hist(bins=50, figsize=(20, 15))
plt.show()


5a
import pandas as pd 
 data = { 'Category': ['A', 'B', 'A', 'C', 'B', 'A'],
'Color': ['Red', 'Blue', 'Green', 'Blue', 'Red', 'Green'], 'Target': [1, 0, 1, 1, 0,
1] } df = pd.DataFrame(data) print(df)
from sklearn.preprocessing import LabelEncoder 
label_encoder = LabelEncoder()
df['Category_LabelEncoded'] = label_encoder.fit_transform(df['Category']) print(df)
df_one_hot_encoded = pd.get_dummies(df, columns=
['Color'], prefix='Color') print(df_one_hot_encoded)
from category_encoders import TargetEncoder
target_encoder = TargetEncoder()
df['Category_TargetEncoded'] = target_encoder.fit_transform(df['Category'],
df['Target']) print(df)
print(df[['Category', 'Category_LabelEncoded',
'Category_TargetEncoded', 'Color', 'Color_Blue', 'Color_Green', 'Color_Red',
'Target']])
import pandas as pd from sklearn.preprocessing import LabelEncoder from
category_encoders import TargetEncoder # Sample data data = { 'Category': ['A', 'B',
'A', 'C', 'B', 'A'], 'Color': ['Red', 'Blue', 'Green', 'Blue', 'Red', 'Green'],
'Target': [1, 0, 1, 1, 0, 1] } df = pd.DataFrame(data) print("Original DataFrame:")
print(df) # Label Encoding label_encoder = LabelEncoder() df['Category_LabelEncoded']
= label_encoder.fit_transform(df['Category']) print("\nLabel Encoded DataFrame:")
print(df) # One-Hot Encoding df_one_hot_encoded = pd.get_dummies(df, columns=
['Color'], prefix='Color') print("\nOne-Hot Encoded DataFrame:")
print(df_one_hot_encoded) # Target Encoding target_encoder = TargetEncoder()
df['Category_TargetEncoded'] = target_encoder.fit_transform(df['Category'],
df['Target']) print("\nTarget Encoded DataFrame:") print(df[['Category',
'Category_LabelEncoded', 'Category_TargetEncoded', 'Color', 'Color_Blue',
'Color_Green', 'Color_Red', 'Target']])


6a
import pandas as pd data = { 'Feature1': [10, 20, 30, 40, 50],
'Feature2': [100, 150, 200, 250, 300] } df = pd.DataFrame(data) print("Original
DataFrame:") print(df)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() 
df_standardized=pd.DataFrame(scaler.fit_transform(df), columns=df.columns) print("\nStandardized
DataFrame:") print(df_standardized)
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler() df_min_max_scaled=
pd.DataFrame(min_max_scaler.fit_transform(df), columns=df.columns) print("\nMin-Max
Scaled DataFrame:") print(df_min_max_scaled)
from sklearn.preprocessing import RobustScaler
robust_scaler = RobustScaler() df_robust_scaled =
pd.DataFrame(robust_scaler.fit_transform(df), columns=df.columns) print("\nRobust
Scaled DataFrame:") print(df_robust_scaled)
import pandas as pd from sklearn.preprocessing import StandardScaler, MinMaxScaler,
RobustScaler  data = { 'Feature1': [10, 20, 30, 40, 50], 'Feature2':
[100, 150, 200, 250, 300] } df = pd.DataFrame(data) print("Original DataFrame:")
print(df) (Z-score Normalization) scaler = StandardScaler()
df_standardized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
print("\nStandardized DataFrame:") print(df_standardized)
min_max_scaler = MinMaxScaler() df_min_max_scaled =
pd.DataFrame(min_max_scaler.fit_transform(df), columns=df.columns) print("\nMin-Max
Scaled DataFrame:") print(df_min_max_scaled) robust_scaler =
RobustScaler() df_robust_scaled = pd.DataFrame(robust_scaler.fit_transform(df),
columns=df.columns) print("\nRobust Scaled DataFrame:") print(df_robust_scaled)


7a
import pandas as pd import numpy as np 
data = { 'Feature1': [1, 2, 3, 4,
5], 'Feature2': [10, 20, 30, 40, 50] } df = pd.DataFrame(data) print("Original
DataFrame:") print(df)
df_log_transformed = df.copy()
df_log_transformed['Feature1'] = np.log1p(df['Feature1'])
df_log_transformed['Feature2'] = np.log1p(df['Feature2']) print("\nLog
Transformed DataFrame:") print(df_log_transformed)
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False
df_poly = poly.fit_transform(df) df_poly =
pd.DataFrame(df_poly, columns=poly.get_feature_names_out(df.columns))
print("\nPolynomial Features DataFrame:") print(df_poly)
from scipy.stats import boxcox
df_boxcox_transformed = df.copy() df_boxcox_transformed['Feature1'], _ =
boxcox(df['Feature1'] + 1)
df_boxcox_transformed['Feature2'], _ = boxcox(df['Feature2'] + 1) print("\nBox-Cox
Transformed DataFrame:") print(df_boxcox_transformed)
import pandas as pd import numpy as np from sklearn.preprocessing import
PolynomialFeatures from scipy.stats import boxcox data = { 'Feature1':
[1, 2, 3, 4, 5], 'Feature2': [10, 20, 30, 40, 50] } df = pd.DataFrame(data)
print("Original DataFrame:") print(df) df_log_transformed =
df.copy() df_log_transformed['Feature1'] = np.log1p(df['Feature1'])
df_log_transformed['Feature2'] = np.log1p(df['Feature2']) print("\nLog Transformed
DataFrame:") print(df_log_transformed) poly =
PolynomialFeatures(degree=2, include_bias=False) df_poly = poly.fit_transform(df)
df_poly = pd.DataFrame(df_poly, columns=poly.get_feature_names_out(df.columns))
print("\nPolynomial Features DataFrame:") print(df_poly)
df_boxcox_transformed = df.copy() df_boxcox_transformed['Feature1'], _ =
boxcox(df['Feature1'] + 1) df_boxcox_transformed['Feature2'], _ =
boxcox(df['Feature2'] + 1) print("\nBox-Cox Transformed DataFrame:")
print(df_boxcox_transformed)
```
