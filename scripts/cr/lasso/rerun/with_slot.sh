#!/bin/bash
# Run a command while holding one of three CPU slot locks, so that all analysis jobs together
# (main queue and the crlasso session) never run more than 3 compute processes on kajman.
# The lock fd (9) is inherited by the exec'd command and its children and is released when
# they exit.
SLOTDIR=/path/to/cr_analysis/queue/slots
mkdir -p "$SLOTDIR"
while true; do
  for i in 1 2 3; do
    exec 9>>"$SLOTDIR/slot$i.lock"
    if flock -n 9; then
      export CR_ANA_SLOT=$i
      echo "$(date -Iseconds) slot$i pid $$ $*" >> "$SLOTDIR/slots.log"
      exec "$@"
    fi
    exec 9>&-
  done
  sleep 15
done
