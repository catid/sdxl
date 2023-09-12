#!/bin/bash

ps a | grep sdxl/server.py | awk '{ print $1 }' | xargs -n1 kill -9 || true
