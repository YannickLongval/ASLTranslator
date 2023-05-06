# ASLTranslator
Detects ASL letters to write words in English, which can then be translated to different languages.

## Setup
After cloning the repository, download the required libraries by running 
```
pip install -r requirements.txt
```
Then, create the model by running <b>train.py</b>. Once the model has finished training and has been saved, <b>main.py</b> can be run to scrape news headlines, and feed them through the model.
