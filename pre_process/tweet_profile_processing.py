import os
import os.path as pth
from tqdm import tqdm
from pre_process.sentence_clean import SentenceClean
from time import sleep
from pre_process.tools import txt2iterable
import pandas as pd

print("--------------------tweet_profile_processing--------------------")

tweet_info_path = pth.join('../datasets/twitter16/raw_data/source_tweet_info.csv')
delete_tweet_list = txt2iterable(pth.join('../datasets/twitter16/auxiliary_data/delete_tweet_list.txt'))

def get_label(lb):
    labelset_nonR, labelset_f, labelset_t, labelset_u = ['news', 'non-rumor'], ['false'], ['true'], ['unverified']
    lb = str(lb).lower()
    if lb in labelset_nonR or lb == '0':
        return 0
    elif lb in labelset_f or lb == '1':
        return 1
    elif lb in labelset_t or lb == '2':
        return 2
    elif lb in labelset_u or lb == '3':
        return 3
    else:
        raise Exception('label not found: {}'.format(lb))


def get_text(sentence) -> "str ('' if None)":
    sentence = str(sentence)
    if sentence is not None:
        sentence = SentenceClean(sentence, method='tokenize').sentence_cleaner()
        return sentence
    else:
        return ''


# source_tweet_info.csv ->  processed_source_tweet.csv
processed_source_tweet_path = pth.join('../load_data16/processed_source_tweet.csv')
print("processed_source_tweet_path: {}".format(processed_source_tweet_path))

content = pd.read_csv(tweet_info_path, lineterminator='\n', encoding='utf-8')
df = pd.DataFrame(content, columns=["source_tweet_id", "source_user_id", "text",
                                    "label", "record_number", "user_number", "unique_tweet_number"])
df = df.where(df.notnull(), None)
df['source_tweet_id'] = df['source_tweet_id'].apply(lambda x: str(x))
df['text'] = df['text'].apply(lambda x: get_text(x))
df['label'] = df['label'].apply(lambda x: get_label(x))
new_df = df    # string in new df
new_df = new_df.rename(columns={"source_tweet_id": "tweet_id", "source_user_id": "user_id"})  # rename column names
new_df.to_csv(processed_source_tweet_path, line_terminator='\n', encoding='utf-8')

print("--source tweet processing finished!--")


# comments processing
tweet_profile_dir = pth.join('../datasets/twitter16/raw_data/tweet_profile')
processed_tree_dir = pth.join('../datasets/twitter16/processed_data/processed_tweet_profile')
if not pth.exists(processed_tree_dir):
    os.mkdir(processed_tree_dir)

delete_tweet_list = [i+'.csv' for i in delete_tweet_list]
available_tweet_list = [i for i in os.listdir(tweet_profile_dir) if i not in set(delete_tweet_list)]
for file in tqdm(available_tweet_list):
    df = pd.read_csv(pth.join(tweet_profile_dir, file), lineterminator='\n')
    df = df.where(df.notnull(), None)
    df = df.rename(columns={'text\r': 'text'})
    df['tweet_id'] = df['tweet_id'].apply(lambda x: str(x))
    # df['user'] = df['user'].apply(lambda x: str(x))
    df['text'] = df['text'].apply(lambda x: get_text(x))
    df2 = df[["tweet_id", "text"]]
    processed_tweet_comments_path = pth.join('../datasets/twitter16/processed_data/processed_tweet_profile', file)
    df2.to_csv(processed_tweet_comments_path, line_terminator='\n', encoding='utf-8')

