import os
import json
import sys
import signal
import flask
from io import StringIO
from functools import reduce

import model_eval as me

# "/home/models/food_not_food_model_v5.tflite"
model_path = "/home/models/2022-03-18_food_not_food_model_efficientnet_lite0_v1.tflite"
model = me.load_model(model_path)

image_home = "/mnt/Data"


# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    # TODO run some health checks.
    status = 200
    image_path = str(Path(image_home).glob("*.jpg").__next__())
    output = model.handle_eval(model, image_path)
    print(f"eval of {image_path} is {output}")
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a batch of data. 
    """
    input_request = None
    request_content_type = flask.request.content_type
    print('DEBUG, flask.request.content_type, "{}"'.format(request_content_type))

    # TODO initialize stuff


    if request_content_type == 'application/json':
        input_request = flask.request.data.decode('utf-8')
        print("input_request", input_request)
        rows = json.loads(input_request)
        print("We got ", len(rows), "inputs")

        # TODO validate input

        results = [model.handle_eval(model, x) for x in rows]

        mimetype = 'application/json'

        result = json.dumps({'result': "ok cool"})
        return flask.Response(response=result, status=200, mimetype=mimetype)


    print('DEBUG, hmm, not application/json')
    return flask.Response(response='This predictor only supports json data',
            status=415, mimetype='text/plain')

