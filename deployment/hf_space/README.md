---
title: Smart Home Keyword Spotting
emoji: 🎙️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
python_version: 3.10
pinned: false
license: cc-by-4.0
---

# Smart Home Keyword Spotting

Speak one supported English command or upload a short WAV file. The model predicts one of 12 labels: ten commands, `unknown`, or `silence`.

Supported commands: yes, no, on, off, stop, go, up, down, left, right.

This model was trained from scratch on a sampled subset of Google Speech Commands v0.02. It may reject uncertain audio. Performance in a live microphone environment can differ from the held-out test set.
