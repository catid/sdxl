#!/bin/bash

ps aux | grep sdxl/server.py | awk '{ print $2 }' | xargs -n1 kill -9 || true
