# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('punkt') # underlying model for tokenization
nltk.download('punkt_tab')

# %%
# the dataset comes from: https://aclanthology.org/2025.wnu-1.1/ (Piper et al. 2025)
df = pd.read_csv("narradetect.csv")
df.head()

# %%
print(df['genre'].value_counts())
print(df['label'].value_counts())
df.tail()
# %%
# sentence tokenize all text
df['sentences'] = df['text'].apply(sent_tokenize)
df['sentlen'] = df['sentences'].apply(len)

# show average sentlen per genre with for-loop
for genre in df['genre'].unique():
    print(f"{genre}: {df[df['genre'] == genre]['sentlen'].mean()}, {df[df['genre'] == genre]['sentlen'].std()}")

# plot it
sns.boxplot(x='genre', y='sentlen', data=df)
plt.title('Sentence Length by Genre')
plt.xticks(rotation=45)
plt.show()

# %%

# get TTR
# tokenize
df['tokens'] = df['text'].apply(word_tokenize) # this is equivalent to [tokenize(text) for text in df['text']]
# get unique tokens
df['types'] = df['tokens'].apply(lambda x: len(set(x)))
df['TTR'] = df['types'] / df['tokens'].apply(len)

# every text already has around 5 sentences
# so we assume that is normalized enough 
# so we don't have to do TTR of a fixed token-length chunk
# but this is an assumption, to be precise we could do fixed chunk

# see TTR per genre
for genre in df['genre'].unique():
    print(f"{genre}: {df[df['genre'] == genre]['TTR'].mean()}, {df[df['genre'] == genre]['TTR'].std()}")

# visualize with dispersion, boxplot
sns.boxplot(x='genre', y='TTR', data=df)
plt.title('Type-Token Ratio by Genre')
plt.xticks(rotation=45)
plt.show()

# %%

# legal vs flash histogram of TTR
plt.figure(figsize=(10, 6))
sns.histplot(df[df['genre'] == 'LEGAL']['TTR'], color='blue', label='LEGAL', kde=True, stat='density')
sns.histplot(df[df['genre'] == 'FLASH']['TTR'], color='orange', label='FLASH', kde=True, stat='density')
plt.title('Type-Token Ratio Distribution for LEGAL vs FLASH')
plt.xlabel('Type-Token Ratio')
plt.ylabel('Density')
plt.legend()

# %%

# same for overall label (narrative/not)
plt.figure(figsize=(10, 6))
sns.histplot(df[df['label'] == 'narrative']['TTR'], color='blue', label='Fiction', kde=True, stat='density')
sns.histplot(df[df['label'] == 'non-narrative']['TTR'], color='orange', label='Nonfiction', kde=True, stat='density')
plt.title('Type-Token Ratio Distribution for Fiction vs Nonfiction')
plt.xlabel('Type-Token Ratio')
plt.ylabel('Density')
plt.legend()



# %%
