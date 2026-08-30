from transformers import GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained("gpt2")

for name, param in model.named_parameters():
    print(name, tuple(param.shape))
    values = param.detach().numpy().flatten()
    with open(f"../weights/{name}.txt", "w") as f:
        f.write(' '.join(str(v) for v in values))

