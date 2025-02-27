#!/bin/bash

sudo git stash push -- restart_proxy.sh
sudo git stash clear
sudo git pull
sudo chmod +x restart_proxy.sh