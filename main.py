import pandas as pd
from nltk import tokenize
from re import sub
from unidecode import unidecode
from multiprocessing import Pool
import math
import string


def tokenizer(tweets):
    tokens = list()
    tk = tokenize.TweetTokenizer(strip_handles=True, reduce_len=True, preserve_case=False)
    for tweet in tweets:
        try:
            element = tk.tokenize(tweet)
        except UnicodeDecodeError:
            element = []
        tokens.append(element)
    return tokens


def normalize(tokens):
    
    exclude = set(string.punctuation)
    # Unicode to string
    tokens_str = [[unidecode(token) for token in tweet] for tweet in tokens]
    # Remove punctuation and numbers
#     tokens_str = [[sub("([,.?!:;-]+|[0-9]+)", "", token) for token in tweet] for tweet in tokens_str]
    tokens_str = [[''.join(ch for ch in token if ch not in exclude) for token in tweet] for tweet in tokens_str]
    
    final_tokens = list()
    for str_list in tokens_str:
        x = filter(None, str_list)
        final_tokens.append(x)
    
    return final_tokens


def createInvertedIndex(tokens):
    inverted_index = {}
    for i in range(len(tokens)):
        for j in range(len(tokens[i])):
            if tokens[i][j]:
               	if inverted_index.has_key(tokens[i][j]):
                	if i not in inverted_index[tokens[i][j]]:
						inverted_index[tokens[i][j]].append(i)
                else:
                    inverted_index[tokens[i][j]] = [i]
        print(i)
    return inverted_index


def bm25(query_tokens, index, tokens):
    
    map_id_score = {}
    k1 = 1.2
    b = 0.75 
    N = len(tokens) # Number of documents in the collection
    adder = 0
    
    for doc_tokens in tokens:
        adder += len(doc_tokens)
    
    avg_doclen = float(adder)/N
    
    for j in range(len(tokens)):
        
        lend = float(len(tokens[j]))
        score = 0
        
        for i in range(len(query_tokens)):
            
            if query_tokens[i] in tokens[j]:
                
                n = float(len(index[query_tokens[i]]))
                f = float(tokens[j].count(query_tokens[i]))
                T1 = math.log(float(N-n+0.5)/(n+0.5),2)
                x = k1 * ((1-b) + b*(lend/avg_doclen)) + f
                T2 = float((k1+1)*f)/x
                score += T1*T2
        
        map_id_score[j] = score
        
    return map_id_score

print("Reading training and testing dataset...")

df_full = pd.read_csv('stanford/training.1600000.processed.noemoticon.csv', header=None)
df_test = pd.read_csv('stanford/testdata.manual.2009.06.14.csv', header=None)

print("Done.")

df = df_full.iloc[:100000]

tweets = list(df[5])

print("Mapping tweetIDs to tweet and sentiment...")

map_id_tweet = {}
map_id_sentiment = {}
for i in range(len(df)):
    map_id_tweet[i] = df.iloc[i][5]
    map_id_sentiment[i] = df.iloc[i][0]

print("Done.")
# pool = Pool()
# tokens = pool.map(tokenizer, tweets)

# Tokenizing the tweeets
print("Tokenizing tweets...")
tokenized_tokens = tokenizer(tweets)
print("Done.")

# Normalizing the tweets
print("Normalizing tweets...")
normalized_tokens = normalize(tokenized_tokens)
print("Done.")

# Creating Inverted Index
print("Creating Inverted Index...")
index = createInvertedIndex(normalized_tokens)
print("Done.")

print("Preprocessing queries...")
queries = list(df_test[5])
tokens1 = tokenizer(queries)
query_tokens = normalize(tokens1)
# print(query_tokens)
print("Done.")

print("\nNumber of tweets in the corpus: " + str(len(df)))
print("Number of index terms: " + str(len(index)))

print("\nPrinting the top 10 results for each query:\n\n")
for i in range(200,250):
    map_id_score = bm25(query_tokens[i], index, normalized_tokens)
    data = map_id_score.items()
    sortedlist = sorted(data, key=lambda x: x[1],reverse=True)
    print("\n\nQuery : " + str(queries[i]))
    print("\nMost Relevant Results : \n")
    for j in range(10):
        print("[" + str(j+1) + "] : (Score = " + str(round(sortedlist[j][1],4)) + ") " + df.iloc[sortedlist[j][0]][5])


# In[163]:

# Removing periods in abbreviations. Ex: U.S.A. to USA

# for tweet in tokens:
#     tweet = [sub(r'(?<!\w)([A-Z])\.', r'\1', x.lower()) for x in tweet]
#     print (tweet)
