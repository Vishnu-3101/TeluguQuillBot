from flask import Flask, request, jsonify
from openai import OpenAI
from flask_cors import CORS
import os
from dotenv import load_dotenv

load_dotenv()

import pandas as pd

app = Flask(__name__)
CORS(app)

# text = "మీరు ఎలా ఉన్నారు?"

client = OpenAI(
          api_key= os.environ['OPENAI_API']
        )
model = "gpt-3.5-turbo"
PARAPHRASE_SYSTEM_PROMPT = f"""You are good at paraphrasing telugu text into telugu. 
For each sentence, give the Paraphrased_sentence: (more formal) in telugu,  a confidence score (How correct the text is).
"""

GRAMMAR_SYSTEM_PROMPT = f"""
You are a telugu grammar checker.
For each sentence that is GRAMMATICALLY incorrect (not anything to do with spelling), give the corrected_sentence (the corrected sentence) in telugu, and an explanation of why your grammar improvement is better in telugu.
"""


@app.route('/predict', methods=['POST'])
def predict():
    data = request.data.decode('utf-8')  # Get the text sent from the frontend
    print(f"Received message from frontend")

    content = GRAMMAR_SYSTEM_PROMPT

    if(data.split(" ")[0].lower()=="paraphrase"):
        content = PARAPHRASE_SYSTEM_PROMPT

    completion = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": content},
                        {"role": "user", "content": data}
                      ]
    )
    paraphrased_text = completion.choices[0].message.content
    print(f"Paraphrased Text")
      # Return the paraphrased text as JSON
    return jsonify({"output": paraphrased_text})
    
    # except Exception as e:
    #     print(e)
    #     return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)