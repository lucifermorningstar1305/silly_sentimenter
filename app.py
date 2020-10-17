"""
created by Адитьям

"""

from flask import Flask, request
from predictor_logistic import predict
from predictor_lstm import main
import json
import traceback


app = Flask(__name__)

@app.route("/", methods=["GET"])
def hello():
	return {"message":"Hello World", "statusCode":200}

@app.route("/predict/logistic/", methods=["GET", "POST"])
def prediction_logistic():
	data = json.loads(request.get_data())
	try:
		results = [predict(v["data"]) for _, v in data.items()]
		print(results)
		return json.dumps({"predicions": results, "statusCode":200})

	except Exception as e:
		print(traceback.print_exc())
		return json.dumps({"message":"Error Occured, please check the logs", "statusCode":500})


@app.route("/predict/lstm/", methods=["GET", "POST"])
def prediction_lstm():
	data = json.loads(request.get_data())
	try:
		results = [main(v["data"]) for _, v in data.items()]
		print(results)
		return json.dumps({"prediction" : results, "statusCode" : 200})

	except Exception as e:
		print(traceback.print_exc())
		return json.dumps({"message":"Error Occured, please check the logs", "statusCode":500})


if __name__ == "__main__":

	app.run(host="0.0.0.0", port=8002, debug=True)