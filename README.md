# Distributed_Inference_For_LLM
Run LLMs using distributed GPU architecture


Generate text with distributed **Llama 3** , **Falcon** (40B+), **BLOOM** (176B) (or their derivatives), and fine‑tune them for your own tasks &mdash; right from your desktop computer or Google Colab:

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

<p align="center">
    <img src="https://i.imgur.com/a/XBj5ueg" width="800">
</p>

How to use:

- Getting started: [tutorial](https://github.com/DataScience-ArtificialIntelligence/Distributed_Inference_For_LLM/blob/main/Inference_On_Public_Swarm.ipynb)

Useful tools:

- [Chatbot web app](https://chat.petals.dev) (connects to Petals via an HTTP/WebSocket endpoint): [source code](https://github.com/petals-infra/chat.petals.dev)
- [Monitor](https://health.petals.dev) for the public swarm: [source code](https://github.com/petals-infra/health.petals.dev)

Advanced guides:

- Launch a private swarm: [guide](https://github.com/bigscience-workshop/petals/wiki/Launch-your-own-swarm)
- Run a custom model: [guide](https://github.com/bigscience-workshop/petals/wiki/Run-a-custom-model-with-Petals)

--------------------------------------------------------------------------------


