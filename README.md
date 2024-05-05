# Distributed_Inference_For_LLM
Run LLMs using distributed GPU architecture


Generate text with distributed **Llama 3** , **Falcon** (40B+), **BLOOM** (176B) (or their derivatives), and fine‑tune them for your own tasks &mdash; right from your desktop computer or Google Colab:

🐧 **Linux + Anaconda.** Run these commands for NVIDIA GPUs (or follow [this](https://github.com/bigscience-workshop/petals/wiki/Running-on-AMD-GPU) for AMD):

```bash
conda install pytorch pytorch-cuda=11.7 -c pytorch -c nvidia
pip install git+https://github.com/bigscience-workshop/petals
python -m petals.cli.run_server petals-team/StableBeluga2
```

🪟 **Windows + WSL.** Follow [this guide](https://github.com/bigscience-workshop/petals/wiki/Run-Petals-server-on-Windows) on our Wiki.

🐋 **Docker.** Run our [Docker](https://www.docker.com) image for NVIDIA GPUs (or follow [this](https://github.com/bigscience-workshop/petals/wiki/Running-on-AMD-GPU) for AMD):

```bash
sudo docker run -p 31330:31330 --ipc host --gpus all --volume petals-cache:/cache --rm \
    learningathome/petals:main \
    python -m petals.cli.run_server --port 31330 petals-team/StableBeluga2
```

🍏 **macOS + Apple M1/M2 GPU.** Install [Homebrew](https://brew.sh/), then run these commands:

```bash
brew install python
python3 -m pip install git+https://github.com/bigscience-workshop/petals
python3 -m petals.cli.run_server petals-team/StableBeluga2
```
## How to Do a sample inference
```python
from transformers import AutoTokenizer
from petals import AutoDistributedModelForCausalLM

# Choose any model available at https://health.petals.dev
model_name = "petals-team/StableBeluga2"  # This one is fine-tuned Llama 2 (70B)

# Connect to a distributed network hosting model layers
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoDistributedModelForCausalLM.from_pretrained(model_name)

# Run the model as if it were on your computer
inputs = tokenizer("A cat sat", return_tensors="pt")["input_ids"]
outputs = model.generate(inputs, max_new_tokens=5)
print(tokenizer.decode(outputs[0]))  # A cat sat on a mat...
```
## How does it work?

- You load a small part of the model, and host a private swarm, and then others can join the network 


![image](https://github.com/DataScience-ArtificialIntelligence/Distributed_Inference_For_LLM/assets/114015823/b63dbb54-e0ca-45b0-87c0-fefe2f8e86e7)


How to use:

- Getting started: [tutorial](https://github.com/DataScience-ArtificialIntelligence/Distributed_Inference_For_LLM/blob/main/Inference_On_Public_Swarm.ipynb)

Useful tools:

- [Chatbot web app](https://chat.petals.dev) (connects to Petals via an HTTP/WebSocket endpoint): [source code](https://github.com/petals-infra/chat.petals.dev)
- [Monitor](https://health.petals.dev) for the public swarm: [source code](https://github.com/petals-infra/health.petals.dev)

Advanced guides:

- Launch a private swarm: [guide](https://github.com/bigscience-workshop/petals/wiki/Launch-your-own-swarm)
- Run a custom model: [guide](https://github.com/bigscience-workshop/petals/wiki/Run-a-custom-model-with-Petals)

--------------------------------------------------------------------------------


