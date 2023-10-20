conda create -n claps python=3.8 -y
conda activate claps
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.24.0
pip install datasets
pip install scipy
pip install sentencepiece