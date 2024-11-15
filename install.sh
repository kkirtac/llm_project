# System dependencies for Pillow
sudo apt update
sudo apt install libjpeg-dev zlib1g-dev libpng-dev libfreetype6-dev

# Create and activate the virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements (excluding torch)
pip install -r requirements.txt

# Install PyTorch and check CUDA availability
pip install torch torchvision torchaudio
python -c "import torch; print(torch.cuda.is_available())"

