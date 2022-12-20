import requests, PIL.Image as Image, io, base64, time

url = "https://tmmqd-01gmd92jc3gfn1cvwasjz61s7c.litng-ai-03.litng.ai/predict"
response = requests.post(url, json={
  "text": "Woman painting a large red egg in a dali landscape"
})
image = Image.open(io.BytesIO(base64.b64decode(response.json()["image"])))
image.save(f"response_{time.time()}.png")
