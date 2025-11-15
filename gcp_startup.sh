#!/bin/bash
apt-get update
apt-get install -y python3 python3-pip git
cd /opt
git clone YOUR_REPO_URL asi
cd asi
pip3 install -r requirements.txt
python3 continuous_asi.py
