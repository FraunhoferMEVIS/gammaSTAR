#!/bin/bash
set -e

cd "$(dirname "$0")"
cwd=$(pwd)
rwd="${cwd/\/home/\/home\/xilinx\/user}"

ssh -o StrictHostKeyChecking=no xilinx@host.docker.internal "bash $rwd/install-host.sh $rwd/docker-compose.yml"
