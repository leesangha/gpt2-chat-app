import os 
import io
import uuid
import shutil
import sys

import threading
import time
from queue import Empty,Queue
import requests

from flask import Flask, render_template,flash,send_file,request,jsonify,url_for
import numpy as np

#####model#####
from dialogpt2 import DialoGPT2
gpt=DialoGPT2(model_name_or_path="microsoft/DialoGPT-small",cuda_device=None,use_context=False)

app=Flask(__name__,template_folder="templates",static_url_path="/static")

url = {
    "tts":"https://master-wave-rnn-woomurf.endpoint.ainize.ai/tts"
}

requests_queue=Queue()
BATCH_SIZE=1
CHECK_INTERVAL=0.1

def handle_requests_by_batch():
    try:
        while True:
            requests_batch=[]
            while not(len(requests_batch) >=BATCH_SIZE):
                try:
                    requests_batch.append(requests_queue.get(timeout=CHECK_INTERVAL))
                except Empty:
                    continue

                for request in requests_batch:
                    request['output']=gpt.gen(request['input'])
    except Exception as e:
        while not requests_queue.empty():
            requests_queue.get()
        print(e)

threading.Thread(target=handle_requests_by_batch).start()

@app.route("/")
def main():
    return render_template("index.html")

@app.route("/read",methods=["POST"])
def read():
    try:
        id=str(uuid.uuid4())
        data={"input_text":"hi","batched":True}
        response=requests.post(url['tts'],data=data)
        if response.status_code==200 :
            file_name = "/app/"+ id + ".wav"
            wav_file = open("/app/"+id+".wav", "wb")
            wav_file.write(response.content)
            print('end')
        return send_file(wav_file,mimetype="audio/wav")
        
    except Exception as e:
        print(e)
        return jsonify({"message":"read wrong"}),429


@app.route("/predict",methods=["POST"])
def predict():
    try:
        if requests_queue.qsize() !=0 :
            print('too many requests')
            return jsonify({"message":"Too Many Requests"}),429
        
        message=request.form["message"]
        req={"input":message}
        requests_queue.put(req)

        while "output" not in req:
            time.sleep(CHECK_INTERVAL)
        if req["output"] == 500:
            return jsonify({"error": "Error! /predict error"}),500
        
        result=req["output"]
        print(result)
        return jsonify({"bot_msg":result}),200
    except Exception as e:
        print(e)
        return jsonify({"message": "Error!"})
@app.route("/health",methods=["GET"])
def health():
    return "ok",200

if __name__ =="__main__":
    from waitress import serve
    serve(app,host="0.0.0.0",port=80)