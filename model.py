from flask import Flask, request, jsonify
from openai import OpenAI
from flask_cors import CORS
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

import pandas as pd

app = Flask(__name__)
CORS(app)

# text = "మీరు ఎలా ఉన్నారు?"

client = Groq(
    api_key=os.environ['GROQ_API_KEY']
)
model = "llama3-8b-8192"
PARAPHRASE_SYSTEM_PROMPT = f"""You are good at paraphrasing telugu text into telugu. 
For each sentence, give the Paraphrased_sentence: (more formal) in telugu,  a confidence score (How correct the text is). For overall paraphrased text give BLEU, ROUGE, METEOR and Cosine Similarity with original sentence.
Format of output: 
Paraphrased Sentence: \n
Confidence Score: \n
BLEU: \n
ROUGE: \n 
METEOR: \n 
Cosine Similarity: \n
"""

GRAMMAR_SYSTEM_PROMPT = f"""
You are a telugu grammar checker.
For each sentence that is GRAMMATICALLY incorrect (not anything to do with spelling), give the corrected_sentence (the corrected sentence) in telugu, and an explanation of why your grammar improvement is better in telugu.
Give GLEU score for corrected sentence.
Output example:
The gramatically corrected sentence is: \n
Explanation: \n
GLEU score:
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