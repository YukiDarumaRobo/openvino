cd openvino
apt update
apt install libgl1 -y
pip install -r requirements.txt
bash omz.sh
python main.py