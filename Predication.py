import json
import requests
import base64

host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=zaz51bMSp3lzDoquRTG0yGx6' \
       '&client_secret=DzsojCUl7U41lEjfbinnj9ifHBYowFjl'
response = requests.get(host)
content = response.json()
access_token = content["access_token"]

image = open('dog.jpg', 'rb').read()
data = {'image': base64.b64encode(image).decode()}

request_url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/classification/dogkinds" + "?access_token=" + access_token
response = requests.post(request_url, data=json.dumps(data))
content = response.json()

print(content.get('results'))
