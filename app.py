from flask import Flask, render_template, request
from transformers import pipeline, AutoTokenizer, DataCollatorWithPadding


def format_usr_input(message):
    percent_encoding_mapping = {
        "%20": " ", "%22": '"', "%25": "%", "%2D": "-", "%2E": ".",
        "%3C": "<", "%3E": ">", "%5C": " \ ", "%5E": "^", "%5F": "_",
        "%60": "`", "%7B": "{", "%7C": "|", "%7D": "}", "%7E": "~",
        "%21": "!", "%23": "#", "%24": "$", "%25": "%", "%26":"&", 
        "%27":"'", "%28": "(", "%29": ")", "%2A": "*", "%2B" :"+",
        "%2C":",", "%3A":":", "%3B":";", "%3D":"=", "%3F": "?",
        "%40": "@", "%5B": "[", "%5D": "]",  "%2F": ""
    }

    msg_new = message.replace("+", " ")
    for key in percent_encoding_mapping.keys():
        msg_new = msg_new.replace(key, percent_encoding_mapping[key])

    return msg_new


app = Flask(__name__, template_folder="templates")

@app.route('/')
def home():
    return render_template('home.html')

#https://www.analyticsvidhya.com/blog/2021/06/a-hands-on-guide-to-containerized-your-machine-learning-workflow-with-docker/#h2_6
@app.route('/',methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
            
        mapping = {"LABEL_1":"Asshole", "LABEL_0": "Not the Asshole"}
        try:
            tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", local_files_only = True)
            model = pipeline("text-classification", model="results/checkpoint-20238/", tokenizer=tokenizer, framework="pt")

            formatted_message = format_usr_input(message)
            prediction = model(formatted_message)
            my_prediction = model(formatted_message)[0]["label"]
            formatted_prediction = mapping[my_prediction]
        except:
            formatted_prediction= "The model could not predict a label for this post. Please try a different post. Shorter posts tend to go through model with less issues."
    return render_template('result.html', prediction = formatted_prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000)

