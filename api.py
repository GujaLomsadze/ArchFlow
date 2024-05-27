"""
Super Basic REST-API endpoint to draw a dynamic Node graph
    and return changed nodes.json to javascript calls.

"""

import json

import redis
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS, cross_origin
from functions import *
from functions.redis_wrapped.json_to_redis import reconstruct_json_from_redis

app = Flask(__name__, static_url_path='/static')
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

redis_connection = redis.Redis(host="localhost", port=6379, decode_responses=True)


# Route to render the HTML template
@app.route('/network/show', methods=['GET'])
def network():
    return render_template('graph_show.html')


# Create an endpoint to handle color updates
@app.route('/update_node_color', methods=['POST'])
def update_node_color():
    data = request.json
    # Implement logic to update node colors based on data['nodeId'] and data['color']
    # You can send a response with a success message or updated data
    response_data = {'success': True, 'message': 'Node color updated'}
    return jsonify(response_data)


@app.route('/graph-analysis')
def graph_analysis():
    return render_template('graph_analysis.html')


@app.route('/get_nodes_n_links')
@cross_origin()
def get_nodes_n_links():
    data = reconstruct_json_from_redis(redis_conn=redis_connection)

    return jsonify(data)


@app.route('/get_current_time')
@cross_origin()
def get_current_time():
    current_time = redis_connection.get("global_timer")
    paths_traversed = redis_connection.get("paths_traversed")

    return jsonify({'time': current_time, "paths_traversed": paths_traversed})


if __name__ == '__main__':
    app.run(debug=True)
