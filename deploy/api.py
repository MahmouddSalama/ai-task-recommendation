import flask
from flask import Flask, request, jsonify
import pandas as pd
import joblib
import json

def skills_tokenizer(x):
        return x
label_encoder= joblib.load("label_encoder.pkl")
task_recommendation_model=joblib.load("task_recommendation_model.pkl")

app = Flask(__name__)


@app.route('/task_recommendation', methods=['POST'])
def api():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No input provided"}), 400
        required_fields = ['Task Type', 'Field', 'Difficulty', 'Current Skills', 
                         'Skills to Learn', 'Available Time per Day (hrs)']
        if not all(field in data for field in required_fields):
            return jsonify({"error": "Missing required fields"}), 400
        
        data['Current Skills'] = ', '.join(sorted(data['Current Skills'].split(', ')))
        data['Skills to Learn'] = ', '.join(sorted(data['Skills to Learn'].split(', ')))
        
        input_df = pd.DataFrame([data])
        
        recommended = task_recommendation_model.predict(input_df)
        
        task = label_encoder.inverse_transform(recommended)
        task_str = task[0] 
        
        
        response = {
            "status": "success",
            "recommended_task": task_str,
            "confidence": None  
        }
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=8000, host='0.0.0.0')