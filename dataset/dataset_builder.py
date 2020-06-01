from sklearn.utils import shuffle

import pandas as pd
import re

from sklearn.model_selection import train_test_split

"""
Pre-process the dataset and split it to train, test and validation datasets.
"""

df = pd.read_json("NTCIR-2020_FinNum_training_v3.json")

for i, row in df.iterrows():
    tweet = row['tweet'].replace(", ", " , ").replace(". ", " . ").replace("\u2019s", " is ")

    numbers = re.findall("\S*\d+\S*", tweet)
    numbers.sort(key=len, reverse=True)
    for index, number in enumerate(numbers):
        tweet = tweet.replace(number, "num{}".format(index))
        if row['target_num'] in number:
            df.at[i, 'target_num'] = "num{}".format(index)

    hashtags = [tag[0] for tag in re.findall("(\$(\w+\.?\w+)?)", tweet)]
    hashtags.sort(key=len, reverse=True)
    for index, hashtag in enumerate(hashtags):
        if len(hashtag) < 2:
            continue
        tweet = tweet.replace(hashtag, "tick{}".format(index))
        ticker = hashtag.replace("$", "")
        tweet = tweet.replace(ticker, "tick{}".format(index))
        if row['target_cashtag'] == ticker:
            df.at[i, 'target_cashtag'] = "tick{}".format(index)
    df.at[i, 'tweet'] = tweet
    if i % 100 == 0:
        print(i)

data = shuffle(df)

train, test = train_test_split(data, test_size=0.1)
train, val = train_test_split(train, test_size=0.1 / 0.9)

train.to_json("train.json", orient='records')
val.to_json("val.json", orient='records')
test.to_json("test.json", orient='records')
