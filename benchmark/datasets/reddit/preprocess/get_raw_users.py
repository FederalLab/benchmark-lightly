# @Author            : FederalLab
# @Date              : 2021-09-26 00:32:44
# @Last Modified by  : Chen Dengsheng
# @Last Modified time: 2021-09-26 00:32:44
# Copyright (c) FederalLab. All rights reserved.
import json
import os
import pickle

from reddit_utils import RedditComment

data_root = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data')

NUM_USERS = 100000
MAX_REPEATS = 500

FILENAME = 'RC_2005-12'
FILE = os.path.join(data_root, 'raw', FILENAME)
SUBS_TO_REMOVE = ['AskReddit', 'ImagesOfNetwork']
USERS_TO_REMOVE = [
    'AskReddit',
    'ImagesOfNetwork',
    'notifier',
    'transcribersofreddit',
    'ilariab95',
    'DJ_Spam',
    'SteamKiwi',
    'AwkwardMod',
    'Mentioned_Videos',
    'MovieGuide',
    'censorship_notifier',
    'MTGCardFetcher',
    'StudabakerHoch',
    'Roboragi',
    'PictureGame',
    '38749grwodsl',
    'one_year_on',
    'koja1234',
    'tippr',
    'User_Simulator',
    'mnemosyne-0002',
    'Decronym',
    'sterlingphoenix',
    'monkey_sage',
    'pussgurka',
    'CosmicKeys',
    'Mafiya_chlenom_K',
    'TheTopazian',
    'nikforce1605',
    'iAdam1n',
    'TitaniumDragon',
    'zeug666',
    '4e4eneK',
    'DoMoreWithLess',
    'Purpleflower88',
    'NeedHelpWithExcel',
    'nicsgalleries',
    'Vkrrs',
    'musono',
    'ZX-Ace',
    'APeskyPanda',
    'TransparentJailbrea',
    'mockingbird',
    'abismado',
    'spotify_transcriber',
]

WORDS_TO_REMOVE = [
    '[deleted]',
    '[removed]',
    'auto',
    'bot',
    'moderator',
    'mod',
    'mentioned',
    'transcriber',
]


def get_users():
    info_dict = {}
    num_repeats = 0
    num_users = 0
    num_total_users = 0
    num_lines = 0

    draft_users_dir = os.path.join(data_root, 'reddit_raw')
    if not os.path.exists(draft_users_dir):
        os.makedirs(draft_users_dir)
    draft_users_file = os.path.join(draft_users_dir, 'reddit_raw_users')

    with open(FILE, 'r') as f:

        # There should be 85 951 976 lines
        for line in f:
            num_lines += 1
            j_line = json.loads(line)

            # I'm assuming subreddit_id and subreddit correspond one to one.
            # A preliminary check with a subsample of the data corroborated this.
            comment = RedditComment(j_line)

            if (any([
                    comment.author.casefold() == u.casefold()
                    for u in USERS_TO_REMOVE
            ]) or any([
                    comment.subreddit.casefold() == s.casefold()
                    for s in SUBS_TO_REMOVE
            ]) or any([
                    w.casefold() in comment.author.casefold()
                    for w in WORDS_TO_REMOVE
            ]) or any([
                    w.casefold() in comment.subreddit.casefold()
                    for w in WORDS_TO_REMOVE
            ])):
                continue

            if comment.author not in info_dict:
                info_dict[comment.author] = []

                num_users += 1
                num_total_users += 1

                if num_total_users % 10000 == 0:
                    print(num_total_users)

            info_dict[comment.author].append(comment)

            if num_users >= NUM_USERS:
                num_repeats += 1
                print()
                print('num_repeats', num_repeats, num_lines)
                print()

                pickle.dump(
                    info_dict,
                    open('{}_{}.pck'.format(draft_users_file, num_repeats),
                         'wb'))

                info_dict = {}
                num_users = 0

            if num_repeats > MAX_REPEATS:
                break

    if num_users > 0 and num_users < NUM_USERS:
        pickle.dump(
            info_dict,
            open('{}_{}.pck'.format(draft_users_file, num_repeats + 1), 'wb'))


def main():
    get_users()


if __name__ == '__main__':
    main()
