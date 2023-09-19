import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ydata_profiling import ProfileReport


csv_path = r'C:\Users\Martín\PycharmProjects\ExploratoryDataAnalysis\supermarket_sales.csv'
df = pd.read_csv(csv_path)
df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)
df.set_index('Date', inplace=True)

# Uni-variate Analysis
# Question 1: What does the distribution of customer ratings looks like? Is it skewed?
sns.histplot(df['Rating'])
plt.axvline(x=np.mean(df['Rating']), c='red', ls='--', label='mean')
plt.axvline(x=np.percentile(df['Rating'], 25), c='green', ls='--', label='25-75th percentile')
plt.axvline(x=np.percentile(df['Rating'], 75), c='green', ls='--')
plt.legend()
plt.show()

df.hist(figsize=(10, 10))

# Question 2: Do aggregate sales numbers differ by much between branches?
custom_colors1 = {'A': 'blue', 'B': 'silver', 'C': 'gold'}
plt.figure(figsize=(10, 8))
sns.countplot(data=df, x='Branch', palette=custom_colors1)

for p in plt.gca().patches:
    plt.gca().annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center',
                       va='baseline',
                       fontsize=12,
                       color='black',
                       xytext=(0, 5),
                       textcoords='offset points')

plt.legend(title='Branch', loc='upper right')
plt.xlabel('Branch')
plt.ylabel('Count')
plt.title('Count of Transactions by Branch')
plt.show()


custom_colors2 = {'Ewallet': 'blue', 'Cash': 'silver', 'Credit card': 'gold'}
plt.figure(figsize=(10, 8))
sns.countplot(data=df, x='Payment', palette=custom_colors2)

for p in plt.gca().patches:
    plt.gca().annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center',
                       va='baseline',
                       fontsize=12,
                       color='black',
                       xytext=(0, 5),
                       textcoords='offset points')

plt.legend(title='Payment', loc='upper right')
plt.xlabel('Payment method')
plt.ylabel('Count')
plt.title('Count of Transactions by Payment method')
plt.show()


# Bi-variate Analysis
# Question 3: Is there a relationship between gross income and customer ratings?
sns.regplot(x='Rating', y='gross income', data=df, scatter_kws={'color': 'blue'}, line_kws={'color': 'gold'})
plt.xlabel('Rating')
plt.ylabel('Gross Income')
plt.title('Rating vs. Gross Income')
plt.show()

custom_colors3 = {'A': 'blue', 'B': 'silver', 'C': 'gold'}
sns.boxplot(x=df['Branch'], y=df['gross income'], palette=custom_colors3)
plt.xlabel('Branch')
plt.ylabel('Gross Income')
plt.title('Gross Income by Branch')
plt.show()


# Question 4: Is there a noticeable time trend in gross income?
plt.figure(figsize=(12, 6))
sns.lineplot(x=df.index, y='gross income', data=df)
plt.title('Gross Income Over Time')
plt.xlabel('Date')
plt.ylabel('Gross Income')
plt.show()

df_ = df.reset_index()
sns.pairplot(data=df_)
plt.show()


# Dealing with duplicate rows and missing values
df.duplicated()
df.duplicated().sum()
df.drop_duplicates(inplace=True)

print(df.isna().sum())

sns.heatmap(df.isnull(), cbar=False)
plt.show()

numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
categorical_columns = df.select_dtypes(include=['object']).columns

df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].mean())
df[categorical_columns] = df[categorical_columns].fillna(df[categorical_columns].mode().iloc[0])

sns.heatmap(df.isnull(), cbar=False)
plt.show()


dataset = pd.read_csv('supermarket_sales.csv')
prof = ProfileReport(dataset)
prof.to_file("report.html")

# Correlation Analysis
numeric_columns = [
    'Unit price',
    'Quantity',
    'Tax 5%',
    'Total',
    'cogs',
    'gross margin percentage',
    'gross income',
    'Rating']

numeric_df = df[numeric_columns]

correlation_matrix = numeric_df.corr().round(2)

sns.heatmap(np.round(numeric_df.corr(), 2), annot=True)
plt.show()
