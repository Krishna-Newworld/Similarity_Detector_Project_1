import requests

url = 'http://127.0.0.1:5000/predict'
data = {
    "sentence1": "There is a cat on the roof.",
    "sentence2": "There is a dog on the roof."
}

response = requests.post(url, json=data)
result = response.json()

print(f"Sentence 1: {result['sentence1']}")
print(f"Sentence 2: {result['sentence2']}")
print("\n🔍 Predictions:")
for model, label in result['predictions'].items():
    print(f"{model}: {label}")
