#!/bin/bash
[ $# -eq 0 ] && echo "Usage: $0 directory" && exit 1
find "$1" -type f -name "*.pthw" -printf "%h\n" | sort -u | while read -r d; do
    find "$d" -maxdepth 1 -type f -name "*.pthw" -printf "%T@ %p\n" | sort -nr | head -n1 | cut -d' ' -f2-
done