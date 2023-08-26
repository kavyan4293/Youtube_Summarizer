# Youtube_Summarizer
Neural Network to Summarize Youtube Video
Have you ever come across a really lengthy Youtube video and thought “If only I had the time and Leisure to watch the entire video“! Was there ever a time when you came across a brand new Gadget Review Video and thought “I wish I could just magically get the crux of the video in a few minutes and save myself some time!“

If your answer is “Yes“ for the above questions, well, I have some good news my dear friend! There is a very Scalable solution that is both accessible and consumable to a wider range of audience.

What is a Summarizer?

A summarizer is a tool or system that automatically condenses and extracts the main points, key ideas, and essential information from a longer piece of text or content, such as an article, document, or video. The goal of a summarizer is to provide a concise and coherent summary that captures the essence of the original content, allowing readers or viewers to quickly grasp the core message without having to go through the entire source material.

How to implement the above?

Step 1 - We will start by installing the required libraries. 

pip install pytube
pip install ffmpeg-python
pip install -U openai-whisper
pip install transformers

Pytube

Python library that provides a convenient way to interact with YouTube videos. It allows you to programmatically download and manipulate YouTube videos using Python code. PyTube simplifies the process of fetching video metadata, downloading video and audio streams, and managing different formats of video content available on YouTube.

Why do we need to install ffmpeg-python when Pytube does all the work?

Because PyTube requires FFmpeg to be installed on our system for certain functionalities. FFmpeg is used by PyTube to process and merge audio and video streams, particularly when we want to download videos or audio from YouTube.

Now what is openai-whisper?

Whisper is an automatic speech recognition (ASR) system trained on 680,000 hours of multilingual and multitask supervised data collected from the web. We show that the use of such a large and diverse dataset leads to improved robustness to accents, background noise and technical language. Moreover, it enables transcription in multiple languages, as well as translation from those languages into English.

Finally, what is one thing that is common in Bumblebee, Megatron and Optimus Prime? 

Transformers! You’re right. But we are using it to summarize the transcribed text. Transformers in Python typically refer to a specific type of deep learning model architecture known as "transformer models." These models have had a significant impact on natural language processing (NLP) and have been used for various tasks such as language translation, text generation, sentiment analysis, and more.

We will be using the Hugging Face Transformers is a popular open-source library and platform that provides easy access to a wide range of pre-trained transformer-based models for natural language processing (NLP) tasks. It was created by Hugging Face, a company focused on democratizing and accelerating AI research and development.

Step 2 - Fetch Youtube Video and extract mp4/mp3 data from the Video using Pytube

from pytube import YouTube
video_url = "https://www.youtube.com/watch?v=Q_aXzpeam0s&list=PL8dPuuaLjXtM6jSpzb5gMNsx9kdmqBfmY&index=13" 
yt = YouTube(video_url)

Import YouTube package from Pytube Library

Store the Youtube video link into a simple variable - video_url

The The YouTube object of Pytube transformer enables us to store the url of the video we need as a String value.

We need to extract only audio codec from our Video and we use “filter“ method. Our video has Dynamic Adaptive Streaming over HTTP (DASH aka Adaptive Stream) and hence we will use the parameter “only_audio“ as True to extract only the mp3/mp4 file from the entire Video link.

We will see how many streams does our Video link have by removing the “first()“ method. This will display all the streams that our DASH video has

video = yt.streams.filter(only_audio=True)

We can see that there are many streams. We will go ahead and extract the first one which is an Audio file with mp4 format -> itag=139

video = yt.streams.filter(only_audio=True).first()

After we’ve selected the stream we’re interested in, we are ready to manipulate with the Audio file by using download() method.

stream.download('',"Youtube_English1.mp4")

Once the above is executed, an mp4 file is stored in your Local Directory.

Step 3 - Transcribe the Youtube Video using Whisper

import whisper
whisper_model = whisper.load_model("base")
result = model.transcribe("Youtube_English1.mp4")

load_model() method uses base model size because our aim is to demonstrate a simple transcribing process here. The bigger the Model size, the more parameters we can use to fine tune our model.

The transcribe() method reads the entire file and processes the audio with a sliding 30-second window, performing autoregressive sequence-to-sequence predictions on each window.

Step 4 - Summarize the Youtube Video using T5 Transformer

from transformers import AutoTokenizer, AutoModelWithLMHead
tokenizer = AutoTokenizer.from_pretrained('t5-small')
model = AutoModelWithLMHead.from_pretrained('t5-small')

Summarization creates a shorter version of a document or an article that captures all the important information. Essentially, there are 2 ways of summarization - Extractive: extract the most relevant information from a document and Abstractive: generate new text that captures the most relevant information.

The Tokenization is taken care by AutoTokenizer which is a generic tokenizing class that will be called upon when used with from_pretrained() method. We need a tokenizer because a NN can only understand digits and it is our responsibility to provide a vector of digits.

The Language Modeling is taken care by AutoModelWithLMHead which is a generic Model class that will be called upon when used with from_pretrained() method as well. This class abstracts away the details of model architecture and loading, making it easier to experiment with different pre-trained models for language generation tasks.

The from_pretrained() method takes care of returning the correct tokenizer class instance based on the type of model we have specified. Here, we specify “t5-small“.

inputs = tokenizer.encode("summarize: " + result['text'], return_tensors='tf', max_length=512, truncation=True)

Let us understand about Encoder-Decoder now!

I came across an example where Encoding-Decoding was compared to playing Pictionary i:e; Dumb Sharades like many Indians can relate more to. 

Encoder - The Person who tries to depict it in a way understandable by the Computer 

Decoder - The Person who tries to understand the clues and hints given by the encoder

Hidden Layer - It is just the way how an encoder explains by giving out clues to the audience and how the decoder(audience) perceives them

In Deep learning T5 model, we have to provide values like “Keyword“, data source, return_tensors, max_length and truncation.

outputs = model.generate(inputs, max_length=100, min_length=20, length_penalty=5., num_beams=2)
summary = tokenizer.decode(outputs)

We generate the Output using generate() method of model library

decode() method simply stores the output in any variable we want.



We can train and finetune the model as follows using T5 transformers using model - AutoModelForSeq2SeqLM

Preprocess - This phase simply consists of prefixing our text with “Summary“ keyword and define the sequence slot length to be taken as inputs. Here we are simply defining a function that will take care of preparing our Input data.

prefix = "summarize: "
checkpoint = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

from transformers import DataCollatorForSeq2Seq
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint, return_tensors="tf")

def preprocess_function(audio_txt):
    inputs = [prefix + doc for doc in audio_txt["text"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)
    labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

preprocess_function(summary)

Train - For this model, we have many hyperparameters to be tuned suring training. We can assign Learning_rate, train and test batch size, train_epochs and evaluation_strategy

training_args = Seq2SeqTrainingArguments(
    output_dir="summary_trained",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=4,
    predict_with_generate=True,
    fp16=True,
)
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=summ["train"],
    eval_dataset=summ["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

Evaluate - ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is a set of metrics commonly used to evaluate the quality of machine-generated text, such as summaries or translations, in comparison to reference human-generated text. ROUGE is often used in the field of natural language processing (NLP) to assess the performance of various text generation tasks, including neural network-based models.

How did we get the User Input - Youtube link URL?

Using a simple Flask App! A Flask app is a web application built using the Flask framework, which is a lightweight and flexible Python web framework. Flask allows you to create web applications quickly and efficiently by providing tools, libraries, and utilities to handle various web-related tasks.

All it requires are 3 folders - 

The static folder containing CSS files, JavaScript files, and images.

The templates folder contains only templates. These have an .html extension and contains simple html files. We have 3 files - getpass, pass and summ_url.html

Lastly, .py files - we have 1 .py file which is named “summarizer.py“

How the repository looks like for a Flask App?



A Template file is required because they are what a User will see on the screen. Here is a simple flowchart of how HTML and .py files communicate with each other.



When we completed building a Flask application with all the abovesaid components, simply deploying this on a 4GB Linux Server Instance enabled us to host publicly on Github. The Website URL receiving page looks something like this:

Once we paste the URL of any Youtube video we like and click on the “search“ button, our code with Pytube, whisper and T5 will run and return the Summary to the same page.



Output:



What can our Summarizer not do for now?

If we want to summarize videos that are more than 2 minutes of length, it starts taking a long time. So, the performance of the model goes for a toss there

The Summary might not be grammatically and punctually correct. Repetition of sentences can be seen

Only Youtube videos are accepted now. Youtube Shorts are not supported