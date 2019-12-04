import praw
from praw.models import MoreComments
fuckBrianWang = []

reddit = praw.Reddit(client_id = 'rHfH895FzPVEOQ',
					client_secret = 'BZoMGE5eEdq_XkFgtX7Is0g7Pu0',
					username='prawWamgg',
					password='Norabbit2!',
					user_agent='brian')

#subreddit = reddit.subreddit('AskReddit')

submissions = reddit.submission(url='https://www.reddit.com/r/AskReddit/comments/1wjy5y/exsmokers_of_reddit_what_actually_worked_to_get/' )

# for submission in submissions.comments(limit=50):
# 	#if(isinstance(submission, MoreComments)):
# 	#	continue
# 	print(submission.body)

# submissions.comments.replace_more(limit=9)
# for comment in submissions.comments.list():
#     print(comment.body)

submissions.comments.replace_more(limit=30)
for comment in submissions.comments.list():
	print(comment.body)
	# fuckBrianWang.append(comment.body)
# print("Brian penis size: ", len(fuckBrianWang), " JK ITS 1mm")
    #print(comment.body)