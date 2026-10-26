import requests

response = requests.post(

    "http://127.0.0.1:8000/detect",

    params={

        "log": "Unauthorized root shell access"

    }

)

print(response.json())