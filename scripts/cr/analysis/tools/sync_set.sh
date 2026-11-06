#!/bin/bash
# sync_set.sh -- copy a read-only input set between two servers through this machine, with sha256 manifests.
#
#   sync_set.sh <src_host> <src_dir> <dst_host> <dst_parent> <name> [--mmin N] [-- <find expression>]
#
# Copies every regular file under <src_host>:<src_dir> that matches the find expression (default: all files)
# to <dst_host>:<dst_parent>/<name>/ (same relative paths) and writes, next to that folder,
#   <dst_parent>/SOURCE_<name>.sha256   hashed on the source host
#   <dst_parent>/<name>.sha256          hashed on the destination host
# The script fails unless the two manifests are identical (so the destination holds exactly the selected
# source files, byte for byte). Re-running it is safe: it re-copies and re-verifies (use it again when the
# source set has grown). --mmin N skips files modified in the last N minutes (a run that is still writing).
# Nothing is written on the source host; on the destination only <dst_parent>/<name>/ and the two manifests.
# Run from the Mac (it streams tar through this machine; the servers do not ssh to each other).
# Paths must not contain spaces or quotes.
set -euo pipefail
if [ $# -lt 5 ]; then sed -n 2,16p "$0"; exit 2; fi
SRC_HOST=$1; SRC_DIR=$2; DST_HOST=$3; DST_PARENT=$4; NAME=$5; shift 5
MMIN=""; EXPR=""
while [ $# -gt 0 ]; do
  case "$1" in
    --mmin) MMIN=$2; shift 2;;
    --) shift; for a in "$@"; do EXPR="$EXPR $(printf '%q' "$a")"; done; break;;
    *) echo "unknown argument $1" >&2; exit 2;;
  esac
done
for p in "$SRC_DIR" "$DST_PARENT" "$NAME"; do
  case "$p" in *[\ \'\"]*) echo "path with a space or quote: $p" >&2; exit 2;; esac
done
SSH=(ssh -o ConnectTimeout=30 -o BatchMode=yes)
FIND="find . -type f"
[ -n "$MMIN" ] && FIND="$FIND -mmin +$MMIN"
[ -n "$EXPR" ] && FIND="$FIND \\( $EXPR \\)"
DST="$DST_PARENT/$NAME"
TMP=$(mktemp -d "${TMPDIR:-/tmp}/sync_set.XXXXXX"); trap 'rm -rf "$TMP"' EXIT

attempt() {   # attempt <description> <function>: up to 5 tries (the network drops)
  local n=1
  until "$2"; do
    [ $n -ge 5 ] && { echo "FAILED: $1" >&2; exit 1; }
    echo "  $1 failed (try $n), waiting 20 s" >&2; n=$((n+1)); sleep 20
  done
}
hash_src() { "${SSH[@]}" "$SRC_HOST" "cd $SRC_DIR && $FIND -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum" > "$TMP/source.sha256"; }
copy()     { "${SSH[@]}" "$SRC_HOST" "cd $SRC_DIR && tar -cf - -T -" < "$TMP/list.txt" |
             "${SSH[@]}" "$DST_HOST" "mkdir -p $DST && chmod -R u+w $DST && cd $DST && tar -xf -"; }
hash_dst() { "${SSH[@]}" "$DST_HOST" "cd $DST && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum" > "$TMP/dest.sha256"; }
put_src()  { "${SSH[@]}" "$DST_HOST" "chmod u+w $DST_PARENT/SOURCE_$NAME.sha256 $DST_PARENT/$NAME.sha256 2>/dev/null; cat > $DST_PARENT/SOURCE_$NAME.sha256" < "$TMP/source.sha256"; }
put_dst()  { "${SSH[@]}" "$DST_HOST" "cat > $DST_PARENT/$NAME.sha256 && chmod a-w $DST_PARENT/SOURCE_$NAME.sha256 $DST_PARENT/$NAME.sha256 && chmod -R a-w $DST" < "$TMP/dest.sha256"; }

echo "[1/4] hashing on $SRC_HOST:$SRC_DIR"
attempt "hashing the source" hash_src
NSRC=$(wc -l < "$TMP/source.sha256" | tr -d ' ')
[ "$NSRC" -gt 0 ] || { echo "no file selected on the source" >&2; exit 1; }
echo "      $NSRC files"
sed -E 's/^[0-9a-f]{64} [ *]//' "$TMP/source.sha256" > "$TMP/list.txt"
echo "[2/4] copying to $DST_HOST:$DST"
attempt "copying" copy
echo "[3/4] hashing on $DST_HOST"
attempt "hashing the destination" hash_dst
if ! cmp -s "$TMP/source.sha256" "$TMP/dest.sha256"; then
  echo "MANIFESTS DIFFER (source vs destination; a file changed during the copy, or the destination holds extra files):" >&2
  diff "$TMP/source.sha256" "$TMP/dest.sha256" | head -20 >&2
  exit 1
fi
echo "[4/4] writing the manifests, making the set read-only"
attempt "writing SOURCE manifest" put_src
attempt "writing destination manifest" put_dst
echo "OK: $NSRC files in $DST_HOST:$DST, manifests identical (sha256 of the manifest $(shasum -a 256 "$TMP/dest.sha256" | cut -d' ' -f1))"
