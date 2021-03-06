# @Author            : FederalLab
# @Date              : 2021-09-26 00:32:36
# @Last Modified by  : Chen Dengsheng
# @Last Modified time: 2021-09-26 00:32:36
# Copyright (c) FederalLab. All rights reserved.
import os
import pickle

from nltk.tokenize import TweetTokenizer

data_root = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data')

DIR = os.path.join(data_root, 'reddit_merged')
FINAL_DIR = os.path.join(data_root, 'reddit_clean')

PHRASES_TO_AVOID = [
    '[ deleted ]',
    '[ removed ]',
    '[deleted]',
    '[removed]',
    'bot',
    'thank you for participating',
    'thank you for your submission',
    'thanks for your submission',
    'your submission has been removed',
    'your comment has been removed',
    'downvote this comment if this is',
    'your post has been removed',
]


def clean_file(f, tknzr):
    reddit = pickle.load(open(os.path.join(DIR, f), 'rb'))

    clean_reddit = {}
    for u, comments in reddit.items():
        clean_comments = []
        for c in comments:
            c.clean_body(tknzr)

            if len(c.body) > 0:
                flag = not any([p in c.body for p in PHRASES_TO_AVOID])
                if flag:
                    clean_comments.append(c)

        if len(clean_comments) > 0:
            clean_reddit[u] = clean_comments

    pickle.dump(
        clean_reddit,
        open(os.path.join(FINAL_DIR, f.replace('merged', 'cleaned')), 'wb'))


def main():
    tknzr = TweetTokenizer()

    if not os.path.exists(FINAL_DIR):
        os.makedirs(FINAL_DIR)

    files = [f for f in os.listdir(DIR) if f.endswith('.pck')]
    files.sort()

    num_files = len(files)
    for i, f in enumerate(files):
        clean_file(f, tknzr)
        print('Done with {} of {}'.format(i, num_files))


if __name__ == '__main__':
    main()
