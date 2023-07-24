import requests

url = 'http://64.247.206.54:5000/generate'
headers = {
    'Content-Type': 'application/json',
    'Accept': 'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
}
data = {
    "prompt": "Write an essay on the book 1989"
}


def print_sse_stream():
    response = requests.post(url, stream=True, headers=headers, json=data)

    print(response.status_code)

    if response.status_code == 200:
        for line in response.iter_content(decode_unicode=True):
            if line:
                print(line)


print_sse_stream()
