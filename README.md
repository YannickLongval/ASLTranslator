# ASLTranslator
Detects ASL letters to write words in English, which can then be translated to different languages.

## Setup
After cloning the repository, download the required libraries by running 
```
pip install -r requirements.txt
```
Then, create the model by running <b>train.py</b>. Once the model has finished training and has been saved, <b>main.py</b> can be run to scrape news headlines, and feed them through the model.
<br/><br/>
To use the translation system, an OpenAI API key must be set up in your environment variables. A guide to getting an API key can be found here: <br/>
[https://www.howtogeek.com/885918/how-to-get-an-openai-api-key/#:~:text=How%20to%20Get%20an%20OpenAI%20API%20Key%201,Secret%20Key%22%20to%20generate%20a%20new%20API%20key.](https://www.howtogeek.com/885918/how-to-get-an-openai-api-key/#:~:text=How%20to%20Get%20an%20OpenAI%20API%20Key%201,Secret%20Key%22%20to%20generate%20a%20new%20API%20key.)
<br/><br/>
After generating your OpenAI API key, create an environment variable called <b>OPENAI_API_KEY</b>, and the API key as the value.
<br/><br/>
Now <b>main.py</b> can be run to detect ASL characters through the webcam.
