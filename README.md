# Strata GLM

**GLM-5.3-Flash (320B-A18B, 157 GB) on one gaming PC**

RTX 5090 + 128 GB RAM + NVMe: **31.3 tokens/s** on a short chat (median of 3; best 32.6), **580 tokens/s** reading a 4,400-token prompt
(llama.cpp on the same PC: 7-10 and 42-55). Images and tool calls work; 262K context.

![Strata GLM against llama.cpp on one RTX 5090 PC: prefill 580 vs 42.3 tokens/s, decode 32.64 vs 9.4 tokens/s](docs/media/strata-glm-huihui-speed.png)

**English** · [日本語](README.ja.md)

> **Unofficial fork, not affiliated with or endorsed by the Strata authors.**

This is a fork of [Strata](https://github.com/Niko1221/Strata) (v0.1.41, MIT) that runs **GLM-5.3-Flash**
(320 billion parameters, 18 billion active, unsloth's **UD-IQ4_XS** GGUF, 157 GB) on **one** PC with a single
consumer GPU, the system RAM and an NVMe drive. Strata itself runs Qwen3.8-Flash-Next; everything GLM-specific
here was added in October 2026 and is kept in this fork. Strata's own README: [README-Strata.md](README-Strata.md).

Nothing leaves your PC. The server speaks the OpenAI and Anthropic APIs, so Claude Code, Codex, opencode,
Cursor and any chat client can talk to it.

## Measured on the development PC

RTX 5090 (32 GB) · Core Ultra 9 285K (24 cores) · 128 GB DDR5-4400 (4 x 32 GB, 2 channels) · NVMe · Windows 11 · 262K context.
"Tokens/s" are measured with `tools/glm5_ref/glm_ab_speed.py`, warm server, 256-token answers.

| | Reads the prompt | Writes the answer |
|---|---:|---:|
| Short chat (86 tokens in) | 1.7 s | **25 tokens/s** |
| After a 560-token document | 5.5 s (100 tokens/s) | 15.5 tokens/s |
| After a 4,400-token document | 7.4 s (590 tokens/s) | 13.3 tokens/s |
| 32K-token prompt | 35-45 s (700-900 tokens/s) | about 8 tokens/s |

The same model with llama.cpp (b11461, best settings found: 16 threads, 40 MoE layers on the CPU) on the same PC
and the same day: prompts at 3-33 tokens/s (86-token prompt 28 s, 4,400 tokens 135 s), answers at 8.1 tokens/s.

The "writes" numbers above use **expert skipping** (below), which changes about one token in a hundred on long
reasoning. With it off the short-chat answer is **20 tokens/s**; the prompt speeds do not depend on it.

### Update 2026-10-14

- **Please update if a question is sometimes misread, especially the first one after a start.** The same user
  reported that with the 2026-10-13 evening build the first answer broke and the second and third did not - thank
  you. That pattern is what the third 2026-10-13 update below fixes (clearing the state before a new conversation
  ran on a stream the computation did not wait for); it was not yet published when they tested. CUDA 13.2 is not
  the cause (measured, see below).
- **The GGUF shards are unmapped once the expert file tier reads unbuffered** (upstream 52a2dd10). With a shard
  mapped, NTFS serves the unbuffered reads of that file one at a time. An 892-token prompt 9.4 -> 7.9 s; the decode
  about 10% faster (same build, interleaved).
- **Lighter host copies in the prompt path.** The prompt's real limit is now the RAM bandwidth: a DMA from page-locked
  RAM makes 46.6 GB/s alone and 17.0 GB/s beside 4 threads running memcpy. The blob copies use non-temporal stores
  (`STRATA_NT_COPY=0`: memcpy) and the blob fetchers are kept across launches (`STRATA_GLM5_PF_CREW=0`: the old
  threads). 4,829 tokens 8.3-8.7 -> 7.5-7.6 s; the decode unchanged.
- **Optional: the expert file tier on two drives** (`STRATA_MIRROR_DIR=<folder>`: a copy of the GGUF shards on a
  second drive; `STRATA_MIRROR_SHARE` = its share of the reads, default 0.5). Each read goes to one of the two, so
  both drives read at once (here 4.0 + 6.9 = 10.9 GB/s). With an 80 GiB RAM tier: start 66 -> 43 s, a short prompt
  4.1 -> 2.4 s, the decode after a 360-token follow-up +15% and after a 7,910-token prompt +25%.
- **Optional: look-ahead in the prompt** (`STRATA_GLM5_PF_AHEAD=256` with `STRATA_GLM5_PREFILL=8192`,
  `--vram-reserve-mib 10000`, `STRATA_GLM5_PF_TOUCH=32`): 7,910 tokens 10.1 -> 8.3 s, but that reserve costs the
  decode 6-7%, and with less reserve the 8192 chunk falls off a cliff (6,400 MiB: 263 s). For long-prompt work only.
- Checked (RTX 5090): over 50 answers correct across the default settings and the two-drive one.

### Update 2026-10-13

- **Fix: answers that were sometimes nothing but "!!!!" (please update).** A user reported that answers broke at
  random and came out right with `STRATA_GLM5_PF_MMQ=0` - thank you. The development PC had shown the same thing on
  2026-10-10 (3 answers in 16 were a run of "!"), left uninvestigated until now.
  - **Cause.** The prompt path sent two small matrices of the KDA layers (`ssm_f_b` and `ssm_g_b`, 128 inputs a row)
    through MMQ (llama.cpp's int8 tensor-core kernels). MMQ reads the weights 256 values at a time, so on a matrix's
    last row it reads **past the matrix**, into VRAM nothing initialized. The activations it meets there are zero, so
    this is usually harmless - but where those bytes happen to form an f16 Inf or NaN scale, the whole answer turns
    into NaN and token 0 ("!") repeats. What lies past the matrix differs from run to run and from PC to PC, hence
    the randomness.
  - **Fix.** MMQ only when the row length is a multiple of 256; otherwise the cuBLAS path as before (the same rule as
    Strata's own `Gemm::native_mmq`).
  - **Checked** (RTX 5090, the example settings, the same 4 prompts): those matrices computed by MMQ from a copy
    whose tail is 0xFF answered "!!!!" to all 4; a 0x00 or 0x3C tail answered all 4 correctly, and so did the fix.
    The prompt time is unchanged (4,829 tokens: 11.6 s before, 11.1-12.1 s after).
  - **`STRATA_GLM5_PF_MMQ=0` is no longer needed** (it also takes the routed experts off MMQ, which slows the prompt).
- **Second fix (evening): saved prompt states are tied to the build (please update, and empty the slot folder
  once).** The same user then reported that the "!!!!" answers were gone but the model still sometimes misread the
  question - thank you again.
  - **The likely cause.** With `STRATA_GLM5_SLOT_DIR` set (it is in the example settings: `build-engine\glm5-slots`), the server keeps a
    prompt's state in files and reads them back after a restart. Those files were matched by model, pack and format,
    **but not by build**: a state written by a build with the over-read above was read by the fixed build, wrong
    numbers and all. Now a slot file of another build is not read, and the start says how many such files are left
    (`... not read (safe to delete)`). We could not reproduce the misreading itself here, so this is the most likely
    cause, not a proven one.
  - **Also fixed.** When a prompt failed with "do not fit" or "page-locked", part of it was already in the state, but
    the server still thought it held the previous conversation; the next request could go on from that. It now starts
    that request over.
  - **Checked** (RTX 5090, one slot folder): the old build wrote a 4,828-token slot; the new build said
    "1 slot file(s) ... not read" and read the prompt again; a second start of the new build read its own slot
    (4,828 tokens in 135 ms). All 12 answers correct.
  - **Not the cause** (measured): CUDA 13.2 vs 12.8 and MMQ on vs off change the next-token probabilities no more
    than running the same settings twice does (mean KL 0.018-0.028 vs 0.014-0.017).
  - If the misreading still happens after this, please open an issue with the prompt.
- **Third update (night): fixes taken from upstream Strata and [Project Maya](https://github.com/mw00/project-maya).**
  - **A new conversation could start from the previous one's state.** Clearing the state, restoring a snapshot and
    reading a slot file ran on a different GPU stream from the computation, which did not wait for them, so the next
    prompt could read the old state while it was still being overwritten. This is another likely cause of the
    misread questions (Maya found the same pattern answering another conversation's question). They now run on the
    compute stream and are waited for.
  - **NaN guard** (after upstream #879): every window's output is checked; a window with NaN / Inf sends none of its
    tokens, the reply ends ("nonfinite"), and the state, the snapshot and the slot files that held it are dropped,
    so the next request starts clean. Its cost is within run-to-run noise. `STRATA_NAN_GUARD=0` turns it off.
  - **GLM's tool tags inside a message are text** (`<tool_call>`, `<arg_key>`, `<arg_value>`, `<tool_response>` and
    their closing tags; upstream #2026 widened as Maya does): a file or a message containing them no longer ends the
    model's tool call or reply.
  - From upstream: a tool list sent in another order keeps its first order (the prompt cache survives, #1624), and a
    keep-alive is sent while a long tool argument is being written (client timeouts, #1666).
  - Checked (RTX 5090): 28 answers correct over 7 starts, including a forced NaN window; the server tests pass (284).

### Update 2026-10-11

- **Tokenizer fix (please update).** The pack's tokenizer split numbers one digit at a time (Qwen3.5's rule);
  GLM-4/5 split them in runs of up to three (llama.cpp's CHATGLM4 rule) and ignore merges for whole words. The
  model then wrote file names and numbers wrong (`2117_00001` came back as `2211_7_0_0001`). Fixed in
  `tools/strata_tokenizer.py`. A pack made before this needs its tokenizer written again:
  `python tools\strata_tokenizer.py --gguf <shard 1 of the GGUF> --out <the pack>\tokenizer`.
  Checked: 12 test texts give the same token ids as llama.cpp's `/tokenize`.
- **Images.** With the model's `mmproj` and `build-vision-glm5.bat` (llama.cpp with the glm5v projector, CPU),
  the server reads pictures (about 414 tokens per image).
- **Tool calls.** GLM's `<tool_call>NAME<arg_key>...` form is parsed as a call (it came back as text before).
- **`--adapt-async 1`** (now in the example settings, with `--adapt-swaps 32`): the VRAM tier's expert swaps run
  on a thread of their own while the next windows compute, instead of between windows (3.5-4.5 ms a window).
  Short chat **28.7-30.1 tokens/s** (24.9-26.0 without it), the follow-up 22.9 (21.5), after a 4,228-token
  summary 20.6 (15-16), prompts unchanged (measured with Huihui's GLM-5.3-Flash abliterated UD-IQ4_XS, whose
  experts are byte-identical to unsloth's; expert skipping on). Which window first uses a swapped-in expert
  depends on the window count only.
- **CPU experts, one token (afternoon).** A one-token window's CPU experts went to ggml's per-token dot instead of
  the AVX-2/AVX-VNNI row kernels (a `max()` with the 2-token default); fixed - an IQ3_S gate/up expert 2.0x faster
  on a P-core (`iq_avx2_parity --pair`). The gate and up rows now run side by side in one loop (the same bits,
  checked for every kernel variant). The pool's phases end sooner: `STRATA_POOL_GUIDED=1` cuts the row ranges
  smaller toward the end and `STRATA_POOL_ETAIL=6` leaves a phase's last 6 tasks to the P-cores (both in the
  example settings; rows are computed alone, so neither changes a bit). Short chat **31.3 tokens/s** (median of 3,
  best 32.6; 29.6 before), the follow-up 24.2 (23.1). After a long summary the speed swings with the experts the
  text routes to (16-21 tokens/s run to run), so that scene does not rank settings.

Quality: the prompt path and every decode optimisation except the skipping are bitwise the engine's exact path
(checked token by token on a deterministic 4,400-token run after each change). The skipping is measured as the
next-token distribution's KL against the exact path: 0.016 on a summary, 0.022 on a 1,213-token reasoning text,
against a run-to-run floor of 0.013 (GPU/CPU rounding).

## What you need

- Windows 10/11 (Linux is untested for the GLM path; the Strata base supports it), Python 3.10+.
- An NVIDIA GPU with 24 GB or more (32 GB tested; 24 GB holds fewer experts in VRAM and will be slower).
  Build tools: Visual Studio 2022 Build Tools (C++ workload) and the CUDA toolkit 12.8 or 13.
- **128 GB of RAM** (the RAM tier holds 98 GB of experts; with 64 GB most experts come from the SSD and decode
  will be several times slower).
- An NVMe SSD with the model on it (the experts not in RAM are read from it, unbuffered) and about 160 GB for the
  GGUF plus 0.3 GB for the pack.
- The model: [unsloth/GLM-5.3-Flash-GGUF](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF), folder
  `UD-IQ4_XS` (5 shards, 157 GB). UD-IQ3_XXS (120 GB) decodes too but its prompts go through the slow path for
  now (its attention weights are Q6_K; see "Known limits").

## Install

```bat
git clone <this repository> strata-glm5
cd strata-glm5
START-HERE.bat --setup          :: Strata's own setup: Python, the .venv (cmake, ninja, numpy, regex ...)
build-glm5.bat                  :: the engine for your GPU (sm_120 for RTX 50; "build-glm5.bat 86" for RTX 30)
.venv\Scripts\python tools\glm5_ref\make_glm5_config.py --gguf D:\models\UD-IQ4_XS\GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf
START-GLM5.bat                  :: http://127.0.0.1:8097/v1  (about a minute to pin the RAM tier)
```

`make_glm5_config.py` builds the pack (`tools/iq_pack.py`: the dense tensors and the expert table, 0.3 GB) in
`Strata-data/packs/glm53_iq4xs` next to the repository, copies the learned expert profile beside it, and writes
`strata-glm5.json` with the RAM tier (total RAM minus 30 GB) and the pool workers (physical cores minus 4) for
your PC. Edit that file for other settings; `strata-glm5.example.json` explains them. Add `--skip` for the expert
skipping (+30% decode, see above).

Strata's `START-HERE.bat --setup` wants to install a Qwen model as well; you can stop it after the Python step
(Ctrl-C once the .venv exists) or let it finish.

### Using it

Point your client at `http://127.0.0.1:8097/v1` (OpenAI: `/v1/chat/completions`, Anthropic: `/v1/messages`),
model `glm-5.3-flash-strata`. For Claude Code / opencode / Cursor see the "Using it" section of Strata's
[Strata's README](README-Strata.md#using-it) (the GLM server behaves the same, on port 8097 instead of 8080).

## How it is fast (and what did not work)

The design notes are in `glm-notes/計画_読み込みの作り替え_2026-10-09.md` (Japanese, with every measurement). The
short version:

**Prompts (prefill).** The experts are *walked layer by layer*: each layer's routed experts are streamed into VRAM
once per group of up to 16K tokens instead of once per 4K chunk, and multiplied on the int8 tensor cores (llama.cpp's
MMQ). The shared experts, the dense layers and the attention projections are tensor-core GEMMs too; the router is one
cuBLAS call; the MLA attention of long prompts runs on WMMA tensor cores with 32 heads a block; the SSD tier is
prefetched a layer ahead; the host state moves in four contiguous arrays while the router runs. 4,400 tokens went from
59.8 s to 7.4 s.

**Answers (decode).** The RAM tier is pinned and ordered by a learned expert profile; the hottest experts live in
VRAM and both tiers follow the conversation (the routing counts of the prompt seed them). What the CPU computes is
read with AVX-VNNI even for one token. The SSD reads of a layer run on a fetcher thread while the RAM-resident experts
compute, and the RAM promotion reads a window ahead. The MLA absorb products of all 64 heads and the KDA projections
that share an input are single launches. Chat decode went from 13.2 to 20.2 tokens/s with the answers unchanged, and
to 25 with the skipping.

**Expert skipping** (`STRATA_GLM5_SKIP_KEEP=2 STRATA_GLM5_SKIP_MINW=0.10`): a token's expert whose routing weight is
under 10% of the token's total, and is not in VRAM, is not computed (the top 2 always are). After the
[2x3090 recipe](https://github.com/saitakarcesme/glm5.3-flash-2x3090)'s cache-aware routing. Off by default.

**Measured and not adopted** (all on this PC): a second PC with an RTX 3060 as a layer stage (14.4 vs 20.2 tokens/s
once the head got fast - the 3060 is 3x slower per layer than the time it saves); computing experts over PCIe
(`--pcie-frac`, the DMA and the CPU share the RAM bus); MTP drafting (the CPU's expert work grows with the window);
a layer-ahead router prediction for the SSD reads (the mispredicted reads cost more than they hide); renormalising the
skipped weight; self-speculation with a few experts (slower even with a perfect drafter); unmapping the GGUF for
faster unbuffered reads (a 15% difference here, the 12 MB reads already saturate the NVMe).

**Where the time goes now** (short chat, 50 ms a token): the CPU's RAM-resident experts 19-25 ms (bandwidth: the
pool reads at 44 GB/s against this PC's 60 GB/s ceiling; faster cores do not help, more memory channels would), the
GPU's ~2,000 small kernel launches 13 ms, the SSD 3 ms, swaps and handshakes the rest.

## Known limits

- One NVIDIA GPU only in this fork's GLM path (the two-PC stage mode is in the code but currently slower; AMD is
  untested).
- UD-IQ3_XXS: decodes (its IQ2_S/IQ3_S experts have CPU and GPU kernels) but prompts need
  `STRATA_GLM5_PREFILL=0` (the verify-window path) because its attention weights are Q6_K.
- The pack is per quantization; `make_glm5_config.py` names it `glm53_<quant>`.
- `tools/glm5_ref/` holds the developer scripts (speed A/B, the quality yardstick `car_follow.py` / `car_kl.py`, the
  Nsight scripts); `mn-test/` is the two-PC experiment. They read the paths from `strata-glm5.json`.

## Credits and license

[Strata](https://github.com/Niko1221/Strata) by Niko1221 and the Strata contributors (MIT); this fork keeps its
license and copyright notice (`LICENSE`). GLM-5.3-Flash by Z.ai; the GGUF by unsloth. llama.cpp's MMQ kernels and
quantization formats (MIT). Ideas measured against or borrowed from [Project Maya](https://github.com/mw00/project-maya),
[glm-flash-lite](https://github.com/sybil-solutions/glm-flash-lite) and the 2x3090 recipe above. The GLM port and
its optimisations were written with Claude (Anthropic) as the pair programmer; every number above was measured on the
development PC and is reproducible with the scripts in `tools/glm5_ref/`.
