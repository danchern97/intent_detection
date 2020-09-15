# flake8: noqa
import pathlib
import json
import os

import requests

SEED = 31415
SERVICE_PORT = int(os.getenv("SERVICE_PORT", 8014))
SERVICE_NAME = os.getenv("SERVICE_NAME", "unknow_skill")


sentences = {
        "sentences":[
            "yes",
            "no",
            "i am good"
            ]
        }


def test_skill():
    url = f"http://0.0.0.0:{SERVICE_PORT}/detect"
    response = requests.post(url, json=sentences).json()
    print(f"Response: {response}")


if __name__ == "__main__":
    test_skill()
