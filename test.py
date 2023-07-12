import requests
url = 'http://localhost:8080/'
headers = {
    'Content-Type': 'application/json',
    'Accept': 'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
}
data = {
    "prompt": "hey there"
}


def print_sse_stream():
    response = requests.post(url, stream=True, headers=headers)

    if response.status_code == 200:
        for line in response.iter_content(decode_unicode=True):
            if line:
                print(line)


print_sse_stream()
