# vLLM version for Qwen/Qwen3.8-27B (researched 2026-11-03)

## Choice: vLLM **v0.32.0** (image `docker://vllm/vllm-openai:v0.32.0`, amd64 9.2 GB compressed)
- Minimum per the vLLM recipe page badge: **0.17.0+** (needs transformers >= 5.8.0). That page also says text
  serving is what it verified; vision is "not covered". The model card itself names no minimum ("use the latest").
- Not clear from the sources which 0.17+ release is the first to run this checkpoint end to end, so: newest stable
  release that clearly supports it = **v0.32.0** (2026-10-18; v0.33.0rc3 is a prerelease, skipped). Evidence:
  - config.json `architectures: ["Qwen3_5ForConditionalGeneration"]`, `model_type: qwen3_5` (Qwen3.8 reuses the
    Qwen3.5 architecture: hybrid linear-attention/full-attention, vision-language, MTP head).
  - vLLM v0.32.0 `registry.py` maps `Qwen3_5ForConditionalGeneration` (and `Qwen3_5MTP`); same in v0.28/0.30/0.31.
  - v0.32.0 `requirements/common.txt` pins `transformers >= 5.16.1, < 5.20.0` (satisfies the recipe's >= 5.8.0).
  - v0.32.0 has the `qwen3_coder` tool parser (`vllm/tool_parsers/__init__.py`) and the `qwen3` reasoning parser
    (`vllm/reasoning/__init__.py`). Checked inside the image (log not included).
- Caveat: 114's image (0.10.1.1) is far too old (no qwen3_5).

## Facts
- Licence: **Apache-2.0** (HF API tag `license:apache-2.0`, `LICENSE` file in repo, model card header).
- Size: 27,781,427,952 BF16 params (HF safetensors metadata) = ~55.6 GB, 18 shards. Dense, not MoE.
- Native context 262,144 (1M via YaRN override; irrelevant here). Vision: images and video (`image-text-to-text`).
- Thinking: **on by default**. Off per request with `"chat_template_kwargs": {"enable_thinking": false}`
  (OpenAI client: `extra_body={"chat_template_kwargs": {"enable_thinking": False}}`). Also `reasoning_effort`
  `xhigh` (default) / `medium` / `low` (chat template argument; recipe passes it as request field), and
  `preserve_thinking` (default true; `chat_template_kwargs`). Reasoning goes to `reasoning`/`reasoning_content`
  only with `--reasoning-parser qwen3`.
- Sampling (card): thinking: temperature 1.0, top_p 0.95, top_k 20, min_p 0, presence_penalty 0, repetition_penalty 1.0.
  Non-thinking: temperature 0.7, top_p 0.80, top_k 20, min_p 0, presence_penalty 1.5, repetition_penalty 1.0.
  `generation_config.json` ships 1.0 / 0.95 / 20, which vLLM uses as server defaults. Card warns that low
  reasoning effort does not always cut total agent latency.

## Serve flags
Recipe default (BF16): `vllm serve Qwen/Qwen3.8-27B --tensor-parallel-size 1 --enable-auto-tool-choice
--tool-call-parser qwen3_coder --reasoning-parser qwen3 --mm-encoder-tp-mode data` (recipe also exports
`VLLM_USE_RUST_FRONTEND=1`; optional, drop it if the frontend misbehaves).
For this project (1 x H200, 141 GB; weights ~56 GB): add `--max-model-len 32768` (task starts at 32768) and
`--served-model-name` as needed. Optional: `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`
(recipe, appended to any command) - not enabled for the first measurement; keep it as a speed experiment.
Do not use `--mm-encoder-tp-mode data` meaning anything at tp=1 (harmless). Image input is OpenAI `image_url`
content (base64 data URL ok); cap with `--limit-mm-per-prompt '{"image":4}'` if `look_around` sends 4 photos.

## Sources
- https://huggingface.co/Qwen/Qwen3.8-27B (card, raw README saved with the check logs, not included)
- https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/config.json (`001-V1-config.json`); HF API `001-V1-hf-model-info.txt`
- https://recipes.vllm.ai/Qwen/Qwen3.8-27B (vLLM recipe: 0.17.0+, transformers >= 5.8.0, flags above)
- https://github.com/vllm-project/vllm/releases/tag/v0.32.0 ; registry/requirements at tag v0.32.0
- Docker Hub tag https://hub.docker.com/r/vllm/vllm-openai/tags (v0.32.0 pushed 2026-10-18)
- Not found/consulted: https://github.com/QwenLM/Qwen3.8 (model card links no repo; recipe + card were sufficient).

## Unverified until the GPU job (0b)
Vision path and tool-call parsing on this exact build; memory/time at tp=1; whether `qwen3_coder` parses this
model's tool output (the card does not name a parser; the recipe does).
