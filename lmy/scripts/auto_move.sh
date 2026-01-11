#!/bin/bash

SRC="processed_dataset"
DST="/var/lib/kubelet/MUYU_data"
N=120        # N 秒内未修改（你可以改）
INTERVAL=60 # 每 60 秒检查一次

mkdir -p "$DST"

while true; do
    find "$SRC" -type f -mmin +$(echo "$N/60" | bc -l) 2>/dev/null | while read -r file; do
        echo "Moving: $file"
        mv "$file" "$DST/"
    done
    sleep "$INTERVAL"
done
