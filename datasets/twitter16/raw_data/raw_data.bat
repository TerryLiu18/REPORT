@echo off
start /wait python tree_checker.py
start /wait python all_user_stat.py
start /wait python tweet2matrix_idx.py
start /wait python source_tweet_info.py
  

