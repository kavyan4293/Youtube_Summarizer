from flask import Flask, jsonify, render_template, request
import whisper
from pytube import YouTube
import torch
from transformers import AutoTokenizer, AutoModelWithLMHead

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('summ_url.html')

@app.route("/val_summ", methods=['GET', 'POST'])
def val_summ():
    if request.method == 'POST':
        url_yutube = request.form
        # Testing to check Post request - return '<h1> The link is : {}</h1>'.format(url_yutube)
        return render_template('pass.html', result = url_yutube)
    else:
        url_yutube = request.args.get('val_summ')
        url_size = request.args.get('val_size')
        url_maxsz = int(url_size)
        if url_maxsz <= 10:
            url_maxsz = 10
            url_minsz = 1
        else:
            url_minsz = url_maxsz - 10
        whisper_model = whisper.load_model("tiny")
        #video_url = "https://www.youtube.com/watch?v=sPwJ0obJya0&list=PL8dPuuaLjXtM6jSpzb5gMNsx9kdmqBfmY&index=1"
        video_url = url_yutube
        yt = YouTube(video_url)
        video = yt.streams.filter(only_audio=True).first()
        stream = yt.streams.get_by_itag(139)
        stream.download('',"GoogleImagen.mp4")
        result = whisper_model.transcribe("GoogleImagen.mp4")
        tokenizer = AutoTokenizer.from_pretrained('t5-small')
        model = AutoModelWithLMHead.from_pretrained('t5-small')

        inputs = tokenizer.encode("summarize: " + result['text'], return_tensors='pt', max_length=512, truncation=True)
        outputs = model.generate(inputs, max_length=url_maxsz, min_length=url_minsz, length_penalty=5., num_beams=2)
        summary = tokenizer.decode(outputs[0])
        return render_template('getpass.html', result = summary)

        ''' # Testing to check get request
        return render_template('getpass.html', result = url_yutube)
        if url_yutube != "":
            #return render_template('pass.html', url_yutube)
        else:'''
        
if __name__ == "__main__":
    app.run()

'''
# Plain version to run to test without any http calls
whisper_model = whisper.load_model("tiny")
video_url = "https://www.youtube.com/watch?v=sPwJ0obJya0&list=PL8dPuuaLjXtM6jSpzb5gMNsx9kdmqBfmY&index=1"
#video_url = url_yutube
yt = YouTube(video_url)
video = yt.streams.filter(only_audio=True).first()
stream = yt.streams.get_by_itag(139)
stream.download('',"GoogleImagen.mp4")
result = whisper_model.transcribe("GoogleImagen.mp4")
tokenizer = AutoTokenizer.from_pretrained('t5-small')
model = AutoModelWithLMHead.from_pretrained('t5-small')

inputs = tokenizer.encode("summarize: " + result['text'], return_tensors='pt', max_length=512, truncation=True)
outputs = model.generate(inputs, max_length=60, min_length=30, length_penalty=5., num_beams=2)
summary = tokenizer.decode(outputs[0])
print(summary)'''
