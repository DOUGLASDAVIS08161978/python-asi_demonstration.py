#!/bin/bash
yum update -y
yum install -y python3 git
cd /opt
git clone YOUR_REPO_URL asi
cd asi
pip3 install -r requirements.txt
nohup python3 continuous_asi.py &
