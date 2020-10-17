# SENTIMENT ANALYSER



### Project Description

This repository contains the code for sentiment analysis using simple Logistic Regression module.

The app for this is hosted in heroku platform. 

The app simply returns whether a review is `positive` or `negative` . There can be many false positive and even false negative also as the test accuracy in 89.95% 

### cURL command for API

Please use this curl command to call the API and then modify the data as per your need.

For using the `Logistic` Model please use this `cURL`

```
curl --location --request POST 'https://sentiment-analyser-linear.herokuapp.com/predict/logistic/' \
--header 'Content-Type: text/plain' \
--data-raw '{"1":{"data":"You are boring me with this problem"}}'
```



For using the `LSTM` Model please use this `cURL`

```
curl --location --request POST 'https://sentiment-analyser-linear.herokuapp.com/predict/lstm/' \
--header 'Content-Type: text/plain' \
--data-raw '{"1":{"data":"You are boring me with this problem"}}'
```

