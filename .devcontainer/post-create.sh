. ../script/setup

sudo apt-get update
sudo apt-get install --yes alsa-utils

groupadd audio || true
usermod -aG audio root