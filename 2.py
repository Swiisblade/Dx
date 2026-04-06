1. Python Object (OOP)
class DataAnalysis:
    def __init__(self, data):
        self.data = data

    def summary(self):
        print(self.data.describe())

analysis = DataAnalysis(df_clean)
analysis.summary()

5. Mean, Std, Percentile
print(df_clean.mean())
print(df_clean.std())
print(df_clean.quantile([0.25,0.5,0.75]))
[4/6, 10:43 AM] Uday (Mca): 7. Frequency Distribution
print(df_clean.iloc[:,0].value_counts())

8. Normal Curve
sns.kdeplot(df_clean.iloc[:,0], fill=True)
plt.show()

9. Hypothesis Testing  0r t-Test
from scipy import stats
t,p = stats.ttest_ind(df_clean.iloc[:,0], df_clean.iloc[:,1])
print(p)

10. Chi-Square Test
from scipy.stats import chi2_contingency
table = pd.crosstab(df_clean.iloc[:,0], df_clean.iloc[:,1])
chi2,p,_,_ = chi2_contingency(table)
print(p)


12. ANOVA
from scipy import stats
f,p = stats.f_oneway(df_clean.iloc[:,0], df_clean.iloc[:,1], df_clean.iloc[:,2])
print(p)

13. Correlation
print(df_clean.corr())

14. Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

X = df_clean.iloc[:,[0]]
y = df_clean.iloc[:,1]

model = LinearRegression().fit(X,y)
pred = model.predict(X)

print(mean_absolute_error(y,pred))
print(mean_squared_error(y,pred))

15. Maximum Likelihood
print(np.mean(df_clean.iloc[:,0]))
print(np.var(df_clean.iloc[:,0]))
20. \K-Means
from sklearn.cluster import KMeans

k = KMeans(n_clusters=3).fit(df_clean)
print(k.labels_)
17. SVM Regression
from sklearn.svm import SVR

model = SVR().fit(X,y)
pred = model.predict(X)

print(mean_absolute_error(y,pred))
 19. SVM Classification
from sklearn.svm import SVC
from sklearn.metrics import classification_report

model = SVC().fit(df_clean,y)
pred = model.predict(df_clean)

print(classification_report(y,pred))
