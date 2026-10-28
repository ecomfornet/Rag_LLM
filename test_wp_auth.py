import requests
import json
import base64

url = "https://kytisa.com/wp-json/wp/v2/posts"
params = {'slug': 'code-general-des-impots-2026-lfi-26'}
username = "agent"
password = "ntbq v9FI kZu5 5wmT VUKC CTI5"

# Try basic auth
res = requests.get(url, params=params, auth=(username, password))
print("Status:", res.status_code)
if res.status_code == 200:
    posts = res.json()
    if posts:
        post = posts[0]
        content = post.get('content', {}).get('rendered', '')
        print("Content length:", len(content))
        print("Content snippet:", content[:200])
    else:
        print("No post found")
else:
    print("Error:", res.text)
