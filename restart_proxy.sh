#!/bin/bash

# sudo chmod +x restart_proxy.sh
# 2 0 * * * /home/ubuntu/freqtrade/restart_proxy.sh >> /home/ubuntu/freqtrade/cron.log 2>&1
# sudo git stash push -- restart_proxy.sh
sudo docker restart binance_proxy_futures