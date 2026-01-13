#!/bin/bash
###
 # @Author: LHL
 # @Date: 2025-07-20 21:11:41
 # @LastEditTime: 2025-08-02 15:48:23
 # @FilePath: /deer-flow/bootstrap.sh
### 

# Start both of DeerFlow's backend and web UI server.
# If the user presses Ctrl+C, kill them both.

if [ "$1" = "--dev" -o "$1" = "-d" -o "$1" = "dev" -o "$1" = "development" ]; then
  echo -e "Starting DeerFlow in [DEVELOPMENT] mode...\n"
  uv run server.py --reload & SERVER_PID=$$!
  cd web && pnpm dev & WEB_PID=$$!
  trap "kill $$SERVER_PID $$WEB_PID" SIGINT SIGTERM
  wait
else
  echo -e "Starting DeerFlow in [PRODUCTION] mode...\n"
  uv run server.py & SERVER_PID=$$!
  cd web && pnpm start & WEB_PID=$$!
  trap "kill $$SERVER_PID $$WEB_PID" SIGINT SIGTERM
  wait
fi
