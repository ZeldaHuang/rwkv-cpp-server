import requests
params={
    "tokenCount":150,
    "temperature":1.2,
    "topP":0.5,
    "presencePenalty":0.4,
    "countPenalty":0.4,
    # "text":"\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese."
    "text":"hello",
}
r = requests.get("http://127.0.0.1:5000/api/chat", params=params)
print(r.content)


r = requests.get("http://127.0.0.1:5000/api/write", params=params)
print(r.content)