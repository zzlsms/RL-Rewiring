# Publishing Checkpoints

Large checkpoints should be hosted in a Hugging Face model repository. GitHub
should contain only source code, documentation, and links to the model page.

## 1. Prepare an inference checkpoint

Upload the directory produced by `save_pretrained`. It should contain model
weights plus the configuration and tokenizer files needed by
`from_pretrained`, for example:

```text
sar-checkpoint/
├── config.json
├── generation_config.json
├── model-00001-of-00002.safetensors
├── model-00002-of-00002.safetensors
├── model.safetensors.index.json
├── tokenizer.json
└── tokenizer_config.json
```

Do not upload credentials, shell history, absolute-path manifests, training
logs, optimizer states, or private datasets. Review the directory before
uploading:

```bash
find /path/to/sar-checkpoint -maxdepth 2 -type f -print
```

## 2. Authenticate on the server

Create a Hugging Face account or organization and authenticate with a token
that has write access:

```bash
hf auth login
hf auth whoami
```

The token is stored in the user's Hugging Face cache. Never place it in this
repository, a shell script, or a model card.

## 3. Upload the checkpoint

Choose a model repository id and upload the checkpoint directory:

```bash
hf upload YOUR_HF_NAMESPACE/sar-checkpoint /path/to/sar-checkpoint .
```

The repository is created automatically when needed. Re-running the command
resumes interrupted large-folder uploads.

Add a model card after replacing its placeholders:

```bash
hf upload YOUR_HF_NAMESPACE/sar-checkpoint docs/model_card_template.md README.md
```

## 4. Link the checkpoint from GitHub

Replace `YOUR_HF_NAMESPACE/sar-checkpoint` in the root `README.md` with the
published model id. Users can then pass the id directly to Transformers or
vLLM without manually downloading the weights.

To materialize a local copy instead:

```bash
hf download YOUR_HF_NAMESPACE/sar-checkpoint \
  --local-dir checkpoints/sar-checkpoint
```

For a private checkpoint, authenticated users can use the same commands. Make
the Hugging Face repository public only after its files and model card have
been reviewed.
