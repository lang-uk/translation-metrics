

# Install Miniconda
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

# Create conda environment
conda create -n metric python=3.11.5 pandas=2.1.1 pytorch=2.1.1 torchvision=0.16.1 pytorch-cuda=11.8 scipy=1.11.3 -c pytorch -c nvidia

# Activate conda environment
conda activate metric

# Install dependencies
# to use GPTQ quantization
pip install auto-gptq==0.5.1
pip install optimum==1.14.1 accelerate==0.24.1
pip install transformers

# to use BnB quantization
pip install bitsandbytes

# to access xCOMET models
pip install --upgrade pip
pip install "unbabel-comet>=2.2.0"
huggingface-cli login               # will have to enter your huggingface access token

# To train distilled models
pip install lightning==2.1.2 wandb

# Visualization
pip install jupyterlab==4.0.9 matplotlib rich

# Onnx for speeding up
pip install onnxruntime

# Workaround for tokenizers
pip install protobuf==3.20