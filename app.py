from flask import Flask, request, render_template, jsonify
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

app = Flask(__name__)

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("./my_model")
model = GPT2LMHeadModel.from_pretrained("./my_model")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = f"###Input: {data['text']}\n###Output:"
    
    encoding = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=100,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,
    )

    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

    # Split by the Output tag to extract only the first completion
    if "###Output:" in decoded_output:
        compressed = decoded_output.split("###Output:")[-1].strip().split("###")[0].strip()
    else:
        compressed = decoded_output.strip()


    return jsonify({'result': compressed})

if __name__ == '__main__':
    app.run(debug=True)
