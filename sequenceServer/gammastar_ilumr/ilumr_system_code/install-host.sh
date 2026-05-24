#!/bin/bash
set -e

echo $1

if [ -e "/etc/matipo/docker-compose.yml" ]; then
    echo "Backup already exists"
else
    echo "Creating backup of default configuration"
    sudo cp /etc/matipo/docker-compose.yml /etc/matipo/docker-compose.yml.default
    echo "Installation failed, check internet connection."
fi

trap 'catch $? $LINENO' ERR

catch() {
    echo "Error occurred in script at line $1"
    echo "Restoring default configuration"

    sudo cp /etc/matipo/docker-compose.yml.default /etc/matipo/docker-compose.yml
    exit 1
}

sudo cp "$1" /etc/matipo/docker-compose.yml
docker compose -f /etc/matipo/docker-compose.yml pull
echo "Docker images pulled"
echo "Press enter to restart, this will take a few minutes and the browser will need to be reloaded."
read -p "[enter]:"

sudo systemctl restart matipo
