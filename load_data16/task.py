"""this is some constant used in the task"""

DATASET_NAME = 'twitter16'
FILTER_NUM = 2
DELAY = 9999

if DATASET_NAME == 'twitter15':
    LOAD_DATA = 'load_data15'
    if DELAY == 10:
    	SOURCE_TWEET_NUM = 1255
    elif DELAY == 20:
    	SOURCE_TWEET_NUM = 1330
    elif DELAY == 30:
    	SOURCE_TWEET_NUM = 1365
    elif DELAY == 60:
    	SOURCE_TWEET_NUM = 1403
    elif DELAY == 90:
        SOURCE_TWEET_NUM = 1419
    elif DELAY == 120:
        SOURCE_TWEET_NUM = 1428
    elif DELAY == 9999:
        SOURCE_TWEET_NUM = 1472
        
if DATASET_NAME == 'twitter16':
    LOAD_DATA = 'load_data16'
    if DELAY == 10:
    	SOURCE_TWEET_NUM = 644
    elif DELAY == 20:
    	SOURCE_TWEET_NUM = 685
    elif DELAY == 30:
    	SOURCE_TWEET_NUM = 709
    elif DELAY == 60:
    	SOURCE_TWEET_NUM = 739
    elif DELAY == 90:
        SOURCE_TWEET_NUM = 744
    elif DELAY == 120:
        SOURCE_TWEET_NUM = 748
    elif DELAY == 9999:  
        SOURCE_TWEET_NUM = 790
# """time parameter"""
# THRESHOLD_TIME = 120   # 10, 20, 30, 60 ,90, 120