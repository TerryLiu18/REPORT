import numpy as np
import re
import pandas as pd
from time import sleep
#
# LANGUAGE = {
#     'en': 'english',
#     'zh': 'chinese'
# }


# def get_url(input_text):
#     urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', input_text)
#     if urls:
#         return urls[0]
#     else:
#         return None

# def replace_url(input_text):
#     text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '<url>', input_text)
#     # if urls:
#     #     return urls[0]
#     # else:
#     return text

# a = 'I love you, https://www.baidu.com'
# b = 'I love you, http://www.baidu.com'
# text = replace_url(a)
# text2 = replace_url(b)
# print(text)
# print(text2)


class SentenceClean:
    def __init__(self,  sentence, hash_tag=True, method='tokenize'):
        self.sentence = sentence
        self.hash_tag = hash_tag  # remain hash_tag
        self.method = method  # use 'regex' or 'hero', default: regex

    def sentence_cleaner(self) -> str:
        """
        :input: sentence, mothed
        :return: dict of words in this sentence
        """
        word_list = []

        if self.sentence is None:
            return None
        if self.method == 'regex':
            """use re to cope with the sentences"""
            y = ' '.join(re.sub("(@[A-Za-z0-9\_\-]+)|([#$&^]+[A-Za-z0-9\_\-]+)|([^0-9A-Za-z\'\- \t])|([' ']+\-[' '])|(\w+:\/\/\S+)", " ", self.sentence).split())
            # y = ' '.join(re.sub("(@[A-Za-z0-9\_\-]+)|([#$&^]+[A-Za-z0-9\_\-]+)|\
            #  ([^0-9A-Za-z\'\- \t])|(\w+:\/\/\S+)", " ", self.sentence).split())
            cleaned_sentence = y.lower()
            # new_word = cleaned_sentence.split()
            # word_list.extend(new_word)
            return cleaned_sentence

        elif self.method == 'keep_punc':
            """use re to cope with the sentences, keep punctuation"""
            y = ' '.join(re.sub("(@[A-Za-z0-9\_\-]+)", " ", self.sentence).split())
            # y = ' '.join(re.sub("(@[A-Za-z0-9\_\-]+)|([#$&^]+[A-Za-z0-9\_\-]+)|\
            #  ([^0-9A-Za-z\'\- \t])|(\w+:\/\/\S+)", " ", self.sentence).split())
            cleaned_sentence = y.lower()
            # new_word = cleaned_sentence.split()
            # word_list.extend(new_word)
            return cleaned_sentence
        elif self.method == 'hero':
            """use texthero to cope with sentences"""
            import texthero as hero
            new_line = self.sentence
            if new_line is None:
                return None
            first_letter = new_line.split()[0][0]
            if first_letter == '@':
                if len(new_line.split()) > 1:
                    new_line = new_line.split(' ', 1)[1]
                else:
                    return None
            line = pd.Series(new_line)
            # print(line)
            line = hero.remove_diacritics(line)
            line = hero.remove_html_tags(line)
            line = hero.remove_urls(line)
            line = hero.remove_whitespace(line)
            line = hero.remove_punctuation(line)
            # print("This is the information after pre-processing\n")
            new_word = str(line).lower()
            new_word = new_word.split(' ')[1:-2]
            print(new_word)
            return new_word
            # print(new_word)
            # word_list.extend(new_word)
            # return word_list
        elif self.method == 'tokenize':
            text = self.sentence

            # replace url in content -> url, change to <url>, whose embedding is available in Glove.twitter
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'url', text)
            text = re.sub("(@[A-Za-z0-9\_\-]+)", " ", text)
            # eliminate #^&*
            text = re.sub("[#*&^~]+", " ", text)
            # Separate most punctuation
            text = re.sub(r"([^\w\.\'\-\/,&])", r' \1 ', text)

            # Separate , and . if they're followed by space.
            # (E.g.,don't separate 2,500)
            text = re.sub(r"(,\s)", r' \1', text)
            text = re.sub(r"([\.]+\s)", r' \1', text)
            text = re.sub(r"(\s[\.]+)", r' \1', text)

            # Separate single quotes
            text = re.sub(r"([\'\"]\s)", ' \' ', text)
            text = re.sub(r"(\s[\'\"])", ' \' ', text)

            # Separate periods that come before newline or end of string.
            text = re.sub('\. *(\n|$)', ' . ', text)
            text = text.lower()
            text = text.replace('\t', ' ')
            text = " ".join(text.split())  # remove duplicate spaces
            text = re.sub('url', '<url>', text)
            text = re.sub('URL', '<url>', text)
            # new_word = text.split()
            return text



if __name__ == '__main__':
    x = ["@Jimmy, mike's 2,500...,2,500,000, 1...2...3, he...me! http://www.google.com #goodluck, ####&*$%^&(",
         "this is really 'insane'!!?\n I couldn't believe it! http://www.google.com",
         "@JasonBWhitman @Lubbockite58 And, scornfully turn WWII vets away.",
         "By all means, let's turn over healthcare decision making to people willingto shut down the Amber Alert website out of spite.",
         "@JasonBWhitman .speaks volumes, if you're willing to listen",
         "@JasonBWhitman @PeteKaliner Wait. Who shut down the Amber Alert website?",
         "@JasonBWhitman @BruceCarrollSC Oh, I doubt the IRS shut down the Amber Alert website.",
         "MT @JasonBWhitman: By all means, let's turn over healthcare decision making to people willing to shut down Amber Alert website out of spite.",
         "MT @hale_razor: @SenTedCruz stood 21 hours for freedom. O**** won't stand for missing children. #AmberAlert #shutdown",
         "@hale_razor LOL. Obama isnt the one who shutdown the govt. Cruz did.",
         "POS petty #Spitehouse RT @hale_razor Cruz stood 21 hours for freedom. Obama won't stand for missing children. #AmberAlert #shutdown",
         "Cruz stood 21 hours for freedom. Obama won't stand for missing children. #AmberAlert #shutdown",
         "One is a Sociopath ""@hale_razor: Cruz stood 21 hours for freedom. Obama won't stand for missing children. #AmberAlert #shutdown”",
         "@hale_razor @matorres87 doesn't it go both ways?",
         "@hale_razor Really? You're fucking retarded. The SITE is down, not the entire amber alert program. Fact check.",
         "Seriously? Racist McDonald's Sign Is Obviously a Hoax - http://on.mash.to/jm5MyY"]
    for s in x:
        print(SentenceClean(s, method='tokenize').sentence_cleaner())

    # print('\n''\n''\n')
    # for s in x:
    #     a = SentenceClean(s, method='hero').sentence_cleaner()

