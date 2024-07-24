# Suicidal Tweet Detection
![Suicidal Tweet Detection](https://snipboard.io/7VKwub.jpg)

This project is a web application that uses machine learning models to classify whether a tweet is a suicide tweet or not. I am using a pretrained DistilBERT model with PyTorch connected to the Flask framework.

## Prerequisites
I used Python version 3.9. You should use that version too! Some previous versions may still be able to run this application.

## Installing
1. Clone this repository.
2. Go to the directory where you cloned this repository.
3. Create a virtual environment on your local system. See: https://flask.palletsprojects.com/en/3.0.x/installation/#virtual-environments.
4. Install all dependencies. To install, run
```
pip install -r requirements.txt
```
6. Run the application with this command
```
python app.py
```
8. You should now be able to access it at http://127.0.0.1:5000/ on your favourite web browser.

## Deployment
Vercel still sucks :( Well, it seems because there are many large dependencies. I am planning to use DigitalOcean instead. Sorry for the inconvenience, but this app can currently only be deployed locally. Will fix it in no time!
