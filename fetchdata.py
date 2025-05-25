import praw
import pprint, json


def get_data(limit : int = 1, subreddit_name="IndiaCricket"):
    reddit = praw.Reddit(
        client_id="86vWzAuA4SzcelcbjWT2Eg",
        client_secret="Pssw9U43Yg-OXixw_tbn2adVkRGiqw",
        password="bu9xiZqNa2vNX56",
        user_agent="python:myredditbot:v1.0 (by u/Popular-Break-6990)",
        username="Popular-Break-6990",
    )

    reddit.read_only = True

    subreddit = reddit.subreddit(subreddit_name)

    def extract_post_attributes(post):
        extracted_data = {
            "id": getattr(post, "id", None),
            "title": getattr(post, "title", None),
            "author": getattr(post, "author", None),
            "subreddit": getattr(post, "subreddit", None),
            "score": getattr(post, "score", None),
            "num_comments": getattr(post, "num_comments", None),
            "created_utc": getattr(post, "created_utc", None),
            "selftext": getattr(post, "selftext", None),
            "topic": getattr(post, "link_flair_text", '').replace(':','').split(),
        }


        return extracted_data

    # content = []
    for submission in subreddit.rising(limit=limit):
        all_comments = submission.comments.list()

        extracted_data = extract_post_attributes(submission)
        comments = []
        for c in all_comments:
            comments.append(c.body)
        extracted_data['comments'] = comments
        extracted_data['author'] = extracted_data['author'].name
        extracted_data['subreddit'] = extracted_data['subreddit'].display_name
        
        # print(extracted_data)
        # content.append(extracted_data)

        with open("reddit_data.json", "w", encoding="utf-8") as f:
            json.dump(extracted_data, f, ensure_ascii=False, indent=2)



