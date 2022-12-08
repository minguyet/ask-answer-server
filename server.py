# Import flask and datetime module for showing date and time
from flask import Flask, request
import datetime
from system import *
 

# Initializing flask app
app = Flask(__name__)


@app.route('/get', methods=['GET'])
def get():
    # print(get_answer(request.json['question']))
    return {
		'Name':"Nguyt",
		"Age":"22",
		"programming":"python"
		}

@app.route('/post', methods=['POST'])
def post():
    ans = {"ans": get_answer(request.json['question'])}
    return ans

# Running app
if __name__ == '__main__':
	load_data()
	print("Load data finish!")
	app.run(debug=True)
