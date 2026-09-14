# DEPLOY-STAGED — streaming capture (`stream-capture` branch) to the live DV4 pair + decoder

> ## ⚠ REQUIRES CHAD'S GO — RESTARTS `vllm_pd` ON BOTH SPARKS (~12 min of no prefill) AND THE FRONT DOOR ON THE DECODER
> Nothing below has been run. The live seat was inspected read-only on 2026-09-14 and is untouched.
> The hook is loaded at container start (`sitecustomize.py` bind-mount), so there is no hot path: the pair must
> come down and back up. The decoder's oMLX server (`:8011`) is **not** touched — only `pd_front.py` (`:8012`)
> restarts (seconds). S2 is never touched.

Placeholders: `SPARK_HEAD` = prefill rank-0 box, `SPARK_WORKER` = rank-1 box, `DECODER` = the Mac running
`pd_front.py`, users as in your `config.env`. Run everything from the box holding this checkout.

## 0. What is live today (inspected 2026-09-14, read-only)

| piece | live | repo `main` | repo `stream-capture` |
|---|---|---|---|
| hook (`~/pd_v3/capture_sitecustomize_v3.py`, bind-mounted as `sitecustomize.py`, **identical md5 on both Sparks**) | v5 + 2026-09-08 survival guards (mem/disk floor, seal, prune, free_layer) | v5 **without** the 09-08 guards | live guards ported + streaming behind `PD_STREAM` |
| container env | `PD_CAPTURE_DIR=/pd_capture PD_CAPTURE_IDLE_S=15.0 PD_POOL_PATH=/pd_v3 PD_PROJ_WEIGHTS=/pd_v3/dv4_proj_weights.safetensors`; engine `--max-model-len 2097152 --max-num-batched-tokens 4096 --gpu-memory-utilization 0.75` | launcher defaults 262144 / 8192 | launcher adds `-e PD_STREAM=${PD_STREAM:-0}` |
| `pd_share.py` (host, `@reboot` cron, `:8010`) | has `/_flush` | **missing `/_flush`** (front calls it) | `/_flush` restored + `/_ack` |
| `pd_front.py` (decoder `~/pd_lab/`, `PD_MODE=pooled`) | 09-07 fixes: `stamp_deadline()`, `PD_LONG_TIMEOUT=10800`; no kv mode | kv mode + `read1` TTFT fix; **flat 900 s timeouts** (would break >800K runs) | both: kv mode + timeouts + `PD_STREAM` consumer |
| `omlx_block_writer.py`, `pd_assemble_blocks.py` | cosmetic diffs only | — | `StreamAssembler` added |
| bench | `~/pd_lab/bench_cold.py` was the 09-06 copy (no `cached_tokens`) | merged version | merged + top-level `verdict`/`transport`; lab copy updated 09-14 |

## 1. Pre-flight (read-only, safe now)

```bash
cd ~/pd_release && git checkout stream-capture && make scrub-check
PY_SPARK=python3 PY_MAC=python3 python3 spark/test_stream_capture.py && python3 studio/test_stream_assembler.py
ssh $SPARK_USER@$SPARK_HEAD   'df -h ~/pd_capture | tail -1; free -g | head -2; docker ps --format "{{.Names}} {{.Status}}"; ss -ltn | grep -E ":8000|:8010"'
ssh $SPARK_USER@$SPARK_WORKER 'docker ps --format "{{.Names}} {{.Status}}"; md5sum ~/pd_v3/capture_sitecustomize_v3.py'
ssh $MAC_USER@$DECODER 'pgrep -fl pd_front.py; curl -s -m 3 localhost:8012/health; curl -s -m 3 localhost:8011/v1/models | head -c 200'
```
Need ≥ 60 GB free in `~/pd_capture`'s filesystem on the head (segments ≈ 10 GB per 1M-token capture, up to 3 kept).

## 2. Stage the files (copies only — nothing restarts yet)

```bash
D=$(date +%Y%m%d-%H%M)
for h in $SPARK_HEAD $SPARK_WORKER; do
  ssh $SPARK_USER@$h "cp ~/pd_v3/capture_sitecustomize_v3.py ~/pd_v3/capture_sitecustomize_v3.py.bak-prestream-$D && cp ~/pd_lab/spark/pd-launch-v3.sh ~/pd_lab/spark/pd-launch-v3.sh.bak-prestream-$D"
  scp spark/capture_sitecustomize_v3.py $SPARK_USER@$h:~/pd_v3/capture_sitecustomize_v3.py.NEW
  scp spark/pd-launch-v3.sh            $SPARK_USER@$h:~/pd_lab/spark/pd-launch-v3.sh.NEW
done
ssh $SPARK_USER@$SPARK_HEAD "cp ~/pd_share.py ~/pd_share.py.bak-prestream-$D"; scp spark/pd_share.py $SPARK_USER@$SPARK_HEAD:~/pd_share.py.NEW
ssh $MAC_USER@$DECODER "cd ~/pd_lab && for f in pd_front.py pd_assemble_blocks.py; do cp \$f \$f.bak-prestream-$D; done"
scp studio/pd_front.py studio/pd_assemble_blocks.py studio/pd_stream_assembler.py $MAC_USER@$DECODER:~/pd_lab/
# pd_stream_assembler.py is new; pd_front.py/pd_assemble_blocks.py are now the repo versions (the decoder's local
# copies differed only by hostnames/timeouts, both of which the repo versions now carry via env / defaults)
```
The live launcher differs from the repo one only by hard-coded per-site paths (HF dir, head IP); **diff before
swapping**: `ssh $SPARK_USER@$SPARK_HEAD 'diff ~/pd_lab/spark/pd-launch-v3.sh ~/pd_lab/spark/pd-launch-v3.sh.NEW'`.
If the live one carries site edits the repo one lacks, add only the one line `-e PD_STREAM=${PD_STREAM:-0}` inside
its `HOOK_MOUNT=(...)` instead of replacing it.

## 3. Deploy — the restart (Chad's go; ~12 min pair reload, front ~30 s)

```bash
# 3a. share (host process, seconds; does not touch vllm_pd). Kill by PORT, never pkill -f.
ssh $SPARK_USER@$SPARK_HEAD 'mv ~/pd_share.py.NEW ~/pd_share.py && fuser -k 8010/tcp; sleep 1;
  setsid nohup python3 ~/pd_share.py ~/pd_capture 8010 >> ~/pd_share.log 2>&1 < /dev/null & sleep 1;
  curl -s -m 3 localhost:8010/_ls | head -c 100; curl -s -m 3 localhost:8010/_flush'
# 3b. hook + launcher into place on both boxes
for h in $SPARK_HEAD $SPARK_WORKER; do ssh $SPARK_USER@$h 'mv ~/pd_v3/capture_sitecustomize_v3.py.NEW ~/pd_v3/capture_sitecustomize_v3.py && mv ~/pd_lab/spark/pd-launch-v3.sh.NEW ~/pd_lab/spark/pd-launch-v3.sh'; done
# 3c. restart the pair WITH THE LIVE ENGINE FLAGS (2097152 window, 4096 chunk) + PD_STREAM=1. Worker first, then head.
ssh $SPARK_USER@$SPARK_WORKER 'PD_HOOK=on PD_STREAM=1 PD_MAX_MODEL_LEN=2097152 PD_CHUNK=4096 PD_CAPTURE_IDLE_S=15.0 bash ~/pd_lab/spark/pd-launch-v3.sh 1'; sleep 5
ssh $SPARK_USER@$SPARK_HEAD   'PD_HOOK=on PD_STREAM=1 PD_MAX_MODEL_LEN=2097152 PD_CHUNK=4096 PD_CAPTURE_IDLE_S=15.0 bash ~/pd_lab/spark/pd-launch-v3.sh 0'
for i in $(seq 1 150); do sleep 10; ssh $SPARK_USER@$SPARK_HEAD 'curl -s -m 3 localhost:8000/v1/models' | grep -q deepseek && break; done
ssh $SPARK_USER@$SPARK_HEAD 'docker logs vllm_pd 2>&1 | grep -a "pd_capture_v3" | grep -a armed | tail -1'   # must say stream=ON
# 3d. front door (decoder): same env it runs with today + PD_STREAM=1. Kill by port.
ssh $MAC_USER@$DECODER 'ps eww -p $(pgrep -f pd_front.py | head -1) | tr " " "\n" | grep -E "^(PD_|OMLX)" > ~/pd_front.env.bak; lsof -ti :8012 | xargs kill; sleep 2;
  cd ~/pd_lab && set -a && . ~/pd_front.env.bak && set +a && PD_STREAM=1 nohup ~/omlx064/bin/python pd_front.py >> ~/pd_front.out 2>&1 & sleep 20; curl -s -m 5 localhost:8012/health'
```

## 4. Verify (in this order; stop at the first red)

```bash
# small cold bridged run: verdict complete, transport tcp10, stream true, cached_tokens ≈ tokens - tail
python3 bench/bench_cold.py --chars 75000 --seed 9101 --url http://$DECODER:8012 | python3 bench/summarize_bench.py -
ssh $SPARK_USER@$SPARK_HEAD 'docker logs vllm_pd 2>&1 | grep -a "stream:" | tail -3; ls ~/pd_capture/*/ | head'   # seg_ files + "stream: N segments"
# medium: 400K, then the 1M that used to seal — watch MemAvailable stay flat on the head while it runs
python3 bench/bench_cold.py --chars 1720000 --seed 9103 --url http://$DECODER:8012 --max-tokens 64 | python3 bench/summarize_bench.py -
ssh $SPARK_USER@$SPARK_HEAD 'for i in $(seq 1 60); do grep MemAvailable /proc/meminfo; sleep 30; done' &   # during the run
python3 bench/bench_cold.py --chars 3900000 --seed 9105 --url http://$DECODER:8012 --max-tokens 64 | python3 bench/summarize_bench.py -
```
Pass = `verdict complete`, `cached_tokens` within one block of `prompt_tokens`, the needle found, and the hook log
showing `STREAM segments=N emitted_T=…` with no `ERROR`. A `partial` at 1M with `flush_reason=mem-floor-seal` means
the backlog estimate is wrong — roll back and bring the log.

## 5. Rollback (same shape, ~12 min)

```bash
for h in $SPARK_HEAD $SPARK_WORKER; do ssh $SPARK_USER@$h "cp ~/pd_v3/capture_sitecustomize_v3.py.bak-prestream-$D ~/pd_v3/capture_sitecustomize_v3.py; cp ~/pd_lab/spark/pd-launch-v3.sh.bak-prestream-$D ~/pd_lab/spark/pd-launch-v3.sh"; done
ssh $SPARK_USER@$SPARK_WORKER 'PD_HOOK=on PD_MAX_MODEL_LEN=2097152 PD_CHUNK=4096 bash ~/pd_lab/spark/pd-launch-v3.sh 1'; sleep 5
ssh $SPARK_USER@$SPARK_HEAD   'PD_HOOK=on PD_MAX_MODEL_LEN=2097152 PD_CHUNK=4096 bash ~/pd_lab/spark/pd-launch-v3.sh 0'
ssh $SPARK_USER@$SPARK_HEAD "cp ~/pd_share.py.bak-prestream-$D ~/pd_share.py; fuser -k 8010/tcp; sleep 1; setsid nohup python3 ~/pd_share.py ~/pd_capture 8010 >> ~/pd_share.log 2>&1 < /dev/null &"
ssh $MAC_USER@$DECODER "cd ~/pd_lab && cp pd_front.py.bak-prestream-$D pd_front.py && cp pd_assemble_blocks.py.bak-prestream-$D pd_assemble_blocks.py; lsof -ti :8012 | xargs kill; sleep 2; set -a; . ~/pd_front.env.bak; set +a; nohup ~/omlx064/bin/python pd_front.py >> ~/pd_front.out 2>&1 &"
```
Half-rollback is also valid: the new hook with `PD_STREAM` unset behaves exactly like the live one (the live 09-08
guards are in it), and the new front with `PD_STREAM` unset runs the one-shot pooled path. **Never run a streaming
hook with a one-shot front** — the layer files then hold only the end state and every bridge declines.

## Downtime budget

| step | who is down | how long |
|---|---|---|
| share restart | capture pulls only | ~2 s |
| pair reload | **all prefill; every bridged request falls back to decoder-native** | ~12 min (measured on window changes) |
| front restart | bridged requests (oMLX `:8011` keeps serving natively) | ~30 s |
| verify runs | shared pair, so nothing else may bench meanwhile | 75K ≈ 2 min · 400K ≈ 7 min · 1M ≈ 20 min |
