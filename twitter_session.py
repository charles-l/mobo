import re

import requests


class TwitterSession:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:78.0) Gecko/20100101 Firefox/78.0"
            }
        )

        # used the twint project to track down these tokens
        req = self.session.prepare_request(
            requests.Request("GET", "https://twitter.com")
        )
        response = self.session.send(req, allow_redirects=True)
        m = re.search(r'\("gt=(\d+);', response.text)
        assert m is not None
        guest_token = str(m.group(1))

        bearer = (
            "Bearer AAAAAAAAAAAAAAAAAAAAANRILgAAAAAAnNwIzUejRCOuH5E6I8xnZz4puTs"
            "%3D1Zv7ttfk8LF81IUq16cHjhLTvJu4FA33AGWWjCpTnA"
        )

        self.headers = {"authorization": bearer, "x-guest-token": guest_token}

    def get_tweet(self, tweet_id):
        r = self.session.get(
            f"https://twitter.com/i/api/2/timeline/conversation/{tweet_id}.json?tweet_mode=extended",
            headers=self.headers,
        )
        return r.json()["globalObjects"]["tweets"][tweet_id]


if __name__ == '__main__':
    from pprint import pprint
    pprint(
        TwitterSession().get("1218527918005063681")["extended_entities"]["media"][0][
            "media_url"
        ]
    )
