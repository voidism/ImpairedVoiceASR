#!/usr/bin/env python
# coding: utf-8
import time
import yaml
import json
import torch
import glob
import shutil
import argparse
import numpy as np
import os
from flask import Flask, request, make_response, Response
from flask import after_this_request
from threading import Thread
import requests
from flask_cors import CORS
# For reproducibility, comment these may speed up training
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Arguments
parser = argparse.ArgumentParser(description='Training E2E asr.')
parser.add_argument('--config', type=str, help='Path to experiment config.')
parser.add_argument('--name', default=None, type=str, help='Name for logging.')
parser.add_argument('--logdir', default='log/', type=str,
                    help='Logging path.', required=False)
parser.add_argument('--logfile', default='log/', type=str,
                    help='Logging path.', required=False)
parser.add_argument('--ckpdir', default='ckpt/', type=str,
                    help='Checkpoint path.', required=False)
parser.add_argument('--outdir', default='result/', type=str,
                    help='Decode output path.', required=False)
parser.add_argument('--load', default=None, type=str,
                    help='Load pre-trained model (for training only)', required=False)
parser.add_argument('--seed', default=0, type=int,
                    help='Random seed for reproducable results.', required=False)
parser.add_argument('--cudnn-ctc', action='store_true',
                    help='Switches CTC backend from torch to cudnn')
parser.add_argument('--njobs', default=6, type=int,
                    help='Number of threads for dataloader/decoding.', required=False)
parser.add_argument('--cpu', action='store_true', help='Disable GPU training.')
parser.add_argument('--no-pin', action='store_true',
                    help='Disable pin-memory for dataloader')
parser.add_argument('--test', action='store_true', help='Test the model.')
parser.add_argument('--demo', action='store_true', help='Demo the model.')
parser.add_argument('--no-msg', action='store_true', help='Hide all messages.')
parser.add_argument('--lm', action='store_true',
                    help='Option for training RNNLM.')
# Following features in development.
parser.add_argument('--amp', action='store_true', help='Option to enable AMP.')
parser.add_argument('--amp_level', default='O2', type=str, help='Option to enable AMP.')
parser.add_argument('--reserve-gpu', default=0, type=float,
                    help='Option to reserve GPU ram for training.')
parser.add_argument('--jit', action='store_true',
                    help='Option for enabling jit in pytorch. (feature in development)')
###
paras = parser.parse_args()
setattr(paras, 'gpu', not paras.cpu)
setattr(paras, 'pin_memory', not paras.no_pin)
setattr(paras, 'verbose', not paras.no_msg)
config = yaml.load(open(paras.config, 'r'), Loader=yaml.FullLoader)

np.random.seed(paras.seed)
torch.manual_seed(paras.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(paras.seed)

# Hack to preserve GPU ram just incase OOM later on server
if paras.gpu and paras.reserve_gpu > 0:
    buff = torch.randn(int(paras.reserve_gpu*1e9//4)).cuda()
    del buff

# Test ASR
assert paras.load is None, 'Load option is mutually exclusive to --demo'
from bin.demo_asr import Solver
mode = 'test'


solver = Solver(config, paras, mode)
# load data for finetune
solver.load_data(batch_size=7)
solver.set_model()

from z2c import z2c

app = Flask(__name__)
CORS(app)

class Finetune(Thread):
    def __init__(self, filename, fixed_text, solver):
        Thread.__init__(self)
        self.filename = filename
        self.fixed_text = fixed_text
        self.solver = solver

    def run(self):
        self.solver.finetune(self.filename, self.fixed_text, max_step=3)

class Clean(Thread):
    def __init__(self):
        Thread.__init__(self)

    def run(self):
        for fname in glob.glob("*.wav"):
            try:
                timestamp = fname.split('.')[0][:-3]
                nowtstamp = time.strftime("%m%d%H%M%S")
                if int(nowtstamp) - int(timestamp) > 1000:
                    os.remove(fname)
                    print("Remove old file %s"%fname, flush=True)
            except:
                print("Not to remove %s" % fname, flush=True)
        return

@app.route("/helloworld", methods=["GET"])
def helloworld():
    return "Hello World!"

@app.route("/log", methods=["GET"])
def log():
    return open(paras.logfile, 'r').read()

@app.route("/tts", methods=["GET"])
def tts():
    url = "http://140.112.29.182:777/tts"
    text = request.args.get('text')
    args = {"text": text}
    r = requests.get(url, params=args)
    return Response(r.content, mimetype="audio/wav")

@app.route("/finetune", methods=["POST"])
def finetune():
    body = json.loads(request.get_data().decode('UTF-8'))
    filename = body["filename"]
    fixed_text = body["text"]
    if '|' in fixed_text:
        fixed_text = fixed_text.split('|')[0]
    # TODO: finetune
    shutil.move(filename, "finetune/new/"+filename)
    _ = os.popen("echo '%s %s' >> finetune/bopomo.trans.txt"%(filename.split('.')[0], fixed_text))
    ft = Finetune("finetune/new/" + filename, fixed_text, solver)
    ft.start()
    return 'finetune!'


@app.route("/recognize", methods=["POST"])
def recognize():

    app.logger.debug(request.files)
    f = request.files["file"]
    save_name = request.form["filename"]
    f.save(save_name)
    
    output = solver.recognize(save_name)
    app.logger.debug(output)
    #return output
    text = z2c(output)
    app.logger.debug(text)
    # Clean old files
    clean = Clean()
    clean.start()
    # @after_this_request
    # def remove_file(response):
    #     for fname in glob.glob("*.wav"):
    #         try:
    #             timestamp = fname.split('.')[0][:-3]
    #             nowtstamp = time.strftime("%m%d%H%M%S")
    #             if int(nowtstamp) - int(timestamp) > 1000:
    #                 os.remove(fname)
    #                 print("Remove old file %s"%fname, flush=True)
    #         except:
    #             print("Not to remove %s"%fname, flush=True)
    #     return response
    return {
        'result': output+'|'+text,
        'filename': save_name
    }

app.run("0.0.0.0", port=1234, debug=True, ssl_context=('./server.crt', './server.key'))
