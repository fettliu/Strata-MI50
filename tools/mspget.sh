#!/bin/bash
# mspget.sh URL OUT START END NCONN [HEADFILE]
# parallel contiguous-range download, resumable.  Parts are concatenated (with an optional
# HEADFILE spliced in front) into OUT.  START/END are absolute byte offsets, END exclusive.
# Re-running skips finished chunks (by size) and a finished OUT (by .mspgot marker).
set -u
URL=$1 OUT=$2 START=$3 END=$4 N=$5 HEAD=${6:-}
TOTAL=$((END-START))
CHUNK=$(( (TOTAL + N - 1) / N ))
[ -f "$OUT.mspgot" ] && [ "$(stat -c%s "$OUT" 2>/dev/null || echo 0)" = "$END" ] \
  && { echo "$OUT complete, skipping"; exit 0; }
echo "fetch $START..$END ($((TOTAL/1048576))MB) in $N chunks of $((CHUNK/1048576))MB -> $OUT"

starts=(); exps=(); files=(); haves=()
for ((i=0;i<N;i++)); do
  s=$((START + i*CHUNK)); e=$((s+CHUNK)); ((e>END)) && e=$END; ((s>=e)) && break
  starts+=($s); exps+=($((e-s))); files+=("$OUT.part$i")
  haves+=($( [ -f "$OUT.part$i" ] && stat -c%s "$OUT.part$i" || echo 0 ))
done

pids=()
for i in "${!starts[@]}"; do
  w=${exps[$i]}; h=${haves[$i]}
  if ((h==w)); then echo "chunk $i done"; pids+=(""); continue; fi
  ((h>w)) && truncate -s $w "${files[$i]}"
  (( $(stat -c%s "${files[$i]}" 2>/dev/null || echo 0) > 0 )) && echo "chunk $i resume from $(stat -c%s "${files[$i]}")/$w"
  (
    st=${starts[$i]} url=$URL f=${files[$i]} want=$w
    for try in 1 2 3 4 5; do
      cur=$(stat -c%s "$f" 2>/dev/null || echo 0)
      ((cur>=want)) && exit 0
      curl -sL --fail --max-time 7200 -r $((st+cur))-$((st+want-1)) "$url" >> "$f"
      sleep 5
    done
    cur=$(stat -c%s "$f" 2>/dev/null || echo 0); ((cur==want))
  ) &
  pids+=($!)
done

for i in "${!pids[@]}"; do
  [ -n "${pids[$i]}" ] || continue
  wait "${pids[$i]}" || echo "chunk $i still short after retries"
done
for i in "${!files[@]}"; do
  got=$(stat -c%s "${files[$i]}" 2>/dev/null || echo 0)
  ((got!=exps[i])) && { echo "chunk $i incomplete $got/${exps[$i]} - rerun me"; exit 1; }
done

{ [ -n "$HEAD" ] && cat "$HEAD"; cat "${files[@]}"; } > "$OUT"
[ "$(stat -c%s "$OUT")" = "$END" ] || { echo "assembled wrong size"; exit 1; }
rm -f "$OUT".part*
touch "$OUT.mspgot"
echo "assembled $(stat -c%s "$OUT") bytes"
