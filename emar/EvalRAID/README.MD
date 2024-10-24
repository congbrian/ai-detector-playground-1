For raid we have several tools

1. generate: create a dataset which will create a dataset based on human info and generate ai info
2. detector: run the detector on the generated data and score it.


run on python 3.10

pip install raid-bench
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124
# the pip assumes you have cuda 12.4 and an nvidia gpu.
pip install tranformers
pip install 'accelerate>=0.26.0'
This does run on windows

