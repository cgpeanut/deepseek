#!/usr/bin/env python3

import ollama
import readline
import argparse
import json
import platform

parser = argparse.ArgumentParser()
#parser.add_argument("model", nargs="?", default="llama3.2")
parser.add_argument("model", nargs="?", default="gdisney/deepseek-coder-uncensored")

args = parser.parse_args()

messages = []

while True:
  try:
    prompt = input(">>> ")
    if prompt == "/bye":
      break
  except:
    print()
    break

  messages.append({"role":"user", "content":prompt})
  response =  ollama.chat(
    args.model,
    messages=messages,
  )

  messages.append(dict(response.message))

  print(response.message.content)

with open("response.log", "a") as f:
  log = {
    "model": args.model,
    "environment": platform.platform(),
    "conversation": messages,
  }
  f.write(json.dumps(log, indent=4))
