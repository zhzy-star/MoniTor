<h1 align="center">
  <img src="assets/fig0.png" width="40" style="vertical-align: middle; margin-right: 10px;" />
  MoniTor: Exploiting Large Language Models with Instruction for Online Video Anomaly Detection
</h1>


<div align="center">

**Shengtian Yang**, **Yue Feng**, **Yingshi Liu**, **Jingrou Zhang**, **Jie Qin**

</div>

<div align="center">
  <img src="https://github.com/user-attachments/assets/4fa0a527-8a79-4778-a62f-beab0a5f94b5" alt="MoniTor Method Overview" width="800">
</div>

> **Abstract:**  
> Video Anomaly Detection (VAD) aims to locate unusual activities or behaviors within videos. Recently, offline VAD has garnered substantial research attention, which has been invigorated by the progress in large language models (LLMs) and vision-language models (VLMs), offering the potential for a more nuanced understanding of anomalies. However, online VAD has seldom received attention due to real-time constraints and computational intensity. In this paper, we introduce a novel Memory-based online scoring queue scheme for Training-free VAD (**MoniTor**), to address the inherent complexities in online VAD. Specifically, MoniTor applies a streaming input to VLMs, leveraging the capabilities of pre-trained large-scale models. To capture temporal dependencies more effectively, we incorporate a novel prediction mechanism inspired by Long Short-Term Memory (LSTM) networks to ensure that the model can effectively model past states and leverage previous predictions to identify anomalous behaviors, thereby better understanding the current frame. Moreover, we design a scoring queue and an anomaly prior to dynamically store recent scores and cover all anomalies in the monitoring scenario, providing guidance for LLMs to distinguish between normal and abnormal behaviors over time.  We evaluate MoniTor on two large datasets (i.e., **UCF-Crime** and **XD-Violence**) containing various surveillance and real-world scenarios. The results demonstrate that MoniTor outperforms state-of-the-art methods and is competitive with weakly supervised methods without training. *Code will be available.*



## Setup

We recommend the use of a Linux machine with CUDA compatible GPUs. We used 2x NVIDIA 4090 GPUs with 24GB. We provide a Conda environment to configure the required libraries.

```bash
git clone https://github.com/YsTvT/MoniTor.git
cd MoniTor
```

## Conda

The environment can be installed and activated with:

```bash
conda create --name Moni python=3.10
conda activate Moni
pip install -r requirements.txt
```
# Data

Please download the data, including captions, temporal summaries, indexes with their textual embeddings, and scores for the UCF-Crime and XD-Violence datasets, from the links below:

- 📦 **UCF-Crime Dataset (processed)**  
  [Download UCF-Crime features](https://drive.google.com/file/d/1_7juCgOoWjQruyH3S8_FBqajuRaORmnV/view)  
  Includes: pre-extracted I3D features, timestamps, textual descriptions, and anomaly labels.

- 📦 **XD-Violence Dataset (processed)**  
  [Download XD-Violence features](https://drive.google.com/file/d/1yzDP1lVwPlA_BS2N5Byr1PcaazBklfkI/view)  
  Includes: temporal annotations, video embeddings, and textual captions.

For the local UCF-Crime test setup, `configs/ucf.json` uses the restored relative
video links under `/root/autodl-tmp/video_add/ucf_crime/videos/test` and the
official filename order from `/root/autodl-tmp/video_add/ucf_crime/splits/test.txt`.
The directory name is `ucf_crime` (underscore). Temporal ground truth is read
from `/root/autodl-tmp/video_add/ucf_crime/annotations/temporal_test.txt`.

Extracted frames and generated artifacts stay in the repository's
`ucf_crime/` directory. Each video gets a frame directory named after its
filename stem, with zero-based `000000.jpg`, `000001.jpg`, etc. The generated
`ucf_crime/annotations/test.txt` records `video_name 0 last_frame_id 0`; the last
frame ID is inclusive. Evaluation derives anomaly labels from temporal ground
truth rather than that placeholder category column.

# Pretrained models

We use **Blip2** as our VLM following [lavad](https://github.com/lucazanella/lavad?tab=readme-ov-file).

The UCF configuration loads all five models from `/root/autodl-tmp/models`:
`blip2-flan-t5-xl`, `blip2-flan-t5-xl-coco`, `blip2-flan-t5-xxl`,
`blip2-opt-6.7b`, and `blip2-opt-6.7b-coco`. Change `model_root` in
`configs/ucf.json` to use another local model directory. Processor and model
loading use `local_files_only=True` for local paths; a missing local directory
raises an error. The captioner also accepts an explicit model directory through
`--pretrained_model_name`.
The processor uses `use_fast=False` so the pinned tokenizer runtime can read
the bundled SentencePiece/BPE vocabularies in the local model folders.

# LLM Agent

We use **GLM-4-Flash** as our default LLM agent due to its efficiency and compatibility.  
You are free to use other models from the **GLM** family depending on your requirements.

🔗 [Visit the official GLM page](https://github.com/THUDM/ChatGLM) to explore available models and documentation.  
🔑 You can obtain your API key from the GLM service at [https://open.bigmodel.cn](https://open.bigmodel.cn).

# Evaluation

Since our method is **training-free Video Anomaly Detection (VAD)**, only the evaluation process is required.

Run stages from the repository root with the active `Moni` environment:

To run all five caption models and the complete UCF test pipeline:

```bash
cd /root/autodl-tmp/MoniTor
conda activate Moni
# ZHIPUAI_API_KEY must already be exported in this shell.
bash scripts/run_ucf_all.sh
```

The script uses `/root/miniconda3/envs/Moni/bin/python` by default, runs stages
sequentially, and writes stage logs under `logs/ucf_<timestamp>_<pid>/`. Configure
the Python executable with `PYTHON` and the JSON config with `UCF_CONFIG`.
The UCF config enables automatic GPU/CPU layer placement with a `20GiB` GPU
weight budget for each BLIP model. This allows larger models to use CPU RAM;
CPU offloading can substantially slow the XXL model.
The UCF pipeline directly follows the reference setup: Top-10 caption cleaning,
10-step ImageBind memory gating, `alpha=0.7`, raw-score queue updates, and no
post-hoc score refinement or three-sample smoothing.

Run an individual stage range, or continue after a completed stage:

```bash
bash scripts/run_ucf_all.sh extract extract
bash scripts/run_ucf_all.sh caption evaluate
bash scripts/run_ucf_all.sh score evaluate
```

Caption, cleaning, summary, memory-embedding, and score stages skip videos with
existing output JSON files. Index stages rebuild their indexes. Extraction
refuses to overwrite an existing generated video list, so start at `caption`
after extraction has completed. If prompts or upstream artifacts change,
select fresh output paths in a copied config to avoid reusing old results.

```bash
conda activate Moni
# Preview the source paths and model commands without executing them.
bash scripts/reproduce.sh --config configs/ucf.json --stage extract --dry-run
bash scripts/reproduce.sh --config configs/ucf.json --stage caption --dry-run

# Run one stage at a time when ready.
bash scripts/reproduce.sh --config configs/ucf.json --stage extract
bash scripts/reproduce.sh --config configs/ucf.json --stage caption --resume
bash scripts/reproduce.sh --config configs/ucf.json --stage caption-index
bash scripts/reproduce.sh --config configs/ucf.json --stage clean --resume
```

Set `ZHIPUAI_API_KEY` in the environment before the summary and scoring stages:

`summary_workers` in `configs/ucf.json` controls concurrent summary API requests
(default UCF configuration: 8). It is independent of the BLIP `batch_size`.
The summary CLI accepts `--workers` (default: 1) and `--max_retries` (default: 4)
for transient connection, rate-limit, and server failures. Empty responses also
receive bounded retries using the same request. Exhausted requests are queued
while other frames and videos continue. Every successful frame is saved atomically
under `caption/summary/_checkpoints/<video_name>.json`, alongside failure metadata.
`--resume` preserves successful frames and only requests missing, retryable frames;
it also validates and skips complete video JSON files. Partial checkpoints verify
the exact input texts, prompt, model, temperature, and sampling interval before reuse.
Keep those settings unchanged when reusing older complete videos.

After the initial sweep, `--recovery_rounds` (default: 3) adds sweeps over retryable
missing frames. `--recovery_delay` (default: 60 seconds) doubles between sweeps,
capped at 300 seconds. Defaults wait 60, 120, then 240 seconds; a persistently
failing frame receives at most 20 requests per invocation (5 per sweep).
Explicit content-filter/refusal responses are recorded as nonretryable and require
manual review; they do not block processing other frames. Authentication, invalid
API parameters, invalid inputs, and disk errors still stop the stage.

Only videos with all expected, nonempty summaries produce final JSON files, ordered
by source frame ID. If any frames remain after recovery, the stage exits nonzero
and the full pipeline stops before scoring; checkpoints retain the missing-frame
queue for inspection and eligible requests can be retried with `--resume`. No empty
or substitute summaries are generated. Per-frame saves add disk writes; additional
failed requests consume time and may consume tokens. Old interrupted runs without
checkpoints cannot recover their unsaved successful summaries. Returned summary
token usage is appended under
`caption/summary/_usage/<video_name>.jsonl`; it covers responses collected by the
new implementation, not historical usage or responses lost during interruption.

`score_workers` controls concurrent UCF video scoring processes (default UCF
configuration: 8; omitted configurations retain 1). Each video keeps its own
memory and score queue, with samples scored sequentially at `batch_size=1`.
This setting uses the scorer's existing `--num_jobs` option and does not load
additional GPU models. `score_request_workers` (default 3) concurrently prepares
each frame's two memory summaries and prediction; `score_prefetch_frames` (default
1) prepares at most the next frame while the current score request is in flight.
The score request and queue updates remain ordered by frame. One scoring client is
used in addition to the preparation clients per video. API request volume is
unchanged; account rate limits can reduce the speedup. GLM uses nonzero temperature,
so concurrent calls preserve the scoring procedure but cannot guarantee identical
responses on a rerun. Scoring retains its existing failure and resume behavior.
With the default 8 video workers, at most 32 API requests can be in flight across
videos. To restore serial request scheduling, set `score_request_workers` to 1 and
`score_prefetch_frames` to 0. Already running processes keep their loaded code.


```bash
bash scripts/reproduce.sh --config configs/ucf.json --stage summarize --resume
bash scripts/reproduce.sh --config configs/ucf.json --stage memory-embeddings --resume
bash scripts/reproduce.sh --config configs/ucf.json --stage score
bash scripts/reproduce.sh --config configs/ucf.json --stage evaluate --score-source raw
```

`--check` validates stage inputs without loading model weights or calling an
API. Extraction preserves the official test split order and writes each source
frame once. It refuses to overwrite an existing generated video list.
Caption JSON files are named `<video_name>.json` with source frame keys
`"0"`, `"16"`, `"32"`, etc. Evaluation writes metrics and aligned frame arrays
under `output/ucf/raw`.

The original checkout omitted the summary conversion stage and complete prompt
files. `src/model/summarize_captions.py` and `prompts/` supply reviewable defaults;
these prompts have not been verified against the paper's exact experimental
prompts. CPU checks validate pipeline behavior, not the reported paper metrics:

```bash
python -m unittest discover -s tests -v
```
