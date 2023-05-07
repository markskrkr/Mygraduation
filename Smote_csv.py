import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



sns.set_theme(style="whitegrid")
df2 = pd.read_csv('D:/毕业设计/faults1.csv')

X = df2.iloc[:,:-1]
y = df2.iloc[:, -1]
from imblearn.over_sampling import SMOTE

# 设置随机数种子
smo = SMOTE(random_state=42)
X_smo, y_smo = smo.fit_resample(X, y)


ss = StandardScaler()
data_ss = ss.fit_transform(X_smo)
df3 = pd.DataFrame(data_ss, columns=X_smo.columns)
df3["Label"] = y_smo
from sklearn.utils import shuffle
df3 = shuffle(df3)
pca = PCA(n_components="mle")
pca_data = pca.fit_transform(df3)
X = df3.drop("Label",axis=1)
y = df3[["Label"]]
processed_data = pd.concat([X, y], axis=1)

processed_data.to_csv('faults_smote.csv', index=False)