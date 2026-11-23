---
name: AMD test result
about: Results of the AMD test build (pre-release amd-test-2, or amd-test-1) on your Radeon card
title: "AMD test: <your card> on <your Linux>"
labels: amd-test
---

Thank you for testing. Please fill in what you can; the self-test output is the most useful part.

**Card** (for example Radeon AI PRO R9700, RX 9070 XT, RX 7900 XTX):

**VRAM** (GB):

**Linux distribution and kernel** (`head -2 /etc/os-release; uname -r`):

**Which tarball** (file name):

**Self-test report** (amd-test-2: run `./selftest.sh` and paste the block from "==== Kintsugi-27B AMD self-test report" to "==== end of report ===="; amd-test-1: `./selftest-amd.sh > selftest.txt 2>&1` and paste the file):

```
paste the report block here
```

**Did the server start and answer?** (amd-test-2: `./start-server.sh --wait`; amd-test-1: `./start-amd.sh`; then a chat request or the browser page)

**Anything else** (errors, speed you saw, what you tried):
