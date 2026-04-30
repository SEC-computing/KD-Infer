#!/bin/bash
######
# Taken from https://github.com/emp-toolkit/emp-readme/blob/master/scripts/throttle.sh
######
## you should authorize this file to be excutable by typing the following command in current directory: chmod 755 netconfig.sh
## to delete the networking setting, type: ./netconfig.sh del
## to simulate a lan, type: ./netconfig.sh lan
## to simulate a wan, type: ./netconfig.sh wan[1/2/3]
## to test the delay, use ping
## to test the bandwidth, use iperf3

## replace DEV=lo with your card (e.g., eth0)
DEV=lo 
if [ "$1" == "del" ]
then
	sudo tc qdisc del dev $DEV root
fi

if [ "$1" == "lan" ]
then
sudo tc qdisc del dev $DEV root
## about 3Gbps
sudo tc qdisc add dev $DEV root handle 1: tbf rate 3000mbit burst 100000 limit 10000
## about 1ms ping latency
sudo tc qdisc add dev $DEV parent 1:1 handle 10: netem delay 0.5msec
fi
if [ "$1" == "wan1" ]
then
sudo tc qdisc del dev $DEV root
## about 400Mbps
sudo tc qdisc add dev $DEV root handle 1: tbf rate 400mbit burst 100000 limit 10000
## about 4ms ping latency
sudo tc qdisc add dev $DEV parent 1:1 handle 10: netem delay 2msec
fi
if [ "$1" == "wan2" ]
then
sudo tc qdisc del dev $DEV root
## about 100Mbps
sudo tc qdisc add dev $DEV root handle 1: tbf rate 100mbit burst 100000 limit 10000
## about 4ms ping latency
sudo tc qdisc add dev $DEV parent 1:1 handle 10: netem delay 2msec
fi
if [ "$1" == "wan3" ]
then
sudo tc qdisc del dev $DEV root
## about 400Mbps
sudo tc qdisc add dev $DEV root handle 1: tbf rate 400mbit burst 100000 limit 10000
## about 40ms ping latency
sudo tc qdisc add dev $DEV parent 1:1 handle 10: netem delay 20msec
fi
