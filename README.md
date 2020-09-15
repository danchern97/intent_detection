# User intent detection in conversation systems.

This is the code provided for Bachelor diploma "User intent detection in conversation systems", 2019, MIPT.

## Deployment
First build a docker container:
`docker build -t intent_catcher .`

Then run it:
`docker run -it -p 8014:8014 intent_catcher`

To send phrases to container, use `python3 test_server.py`. To modify phrases, see `test_server.py` and modify the request sentences.

Intent data is located at `data/full_dataset.json` at `data['alexa_prize']['train']`. The list of intents is located there.
