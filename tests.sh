#!/bin/bash
# Verification sweep per docs/GPU_DEVELOPMENT.md section 4.
# Usage: ./tests.sh [binary] [extra args, e.g. --gpu]
#
# Path-activation thresholds this gate must stay past (any path with an
# activation threshold needs >= 1 point beyond it here or it will rot):
#   * pend migration bug .......... first-multiple offset exceeds one segment
#     for p > 30B/7 ~ 1.1M, i.e. n > ~1.3e12  -> windows at 2e12..2e13 below
#   * bucket sieve ................ first engages for p > 3B ~ 768K
#   * countLo/cap on a segment edge (phantom candidate off-by-one) -> the
#     k*span edge windows below (span = 30*B = 7864320 at the default B)
set -u
BIN=${1:-./fastsieve}
if [ $# -gt 1 ]; then shift; fi
PS=${PRIMESIEVE:-$HOME/.local/bin/primesieve}
pass=0; fail=0

check() { # name got want
  if [ "$2" == "$3" ]; then
    echo "PASS $1: $2"; pass=$((pass+1))
  else
    echo "FAIL $1: got $2 want $3"; fail=$((fail+1))
  fi
}

for n in 100 1000 10000 100000 1000000 10000000 100000000 1000000000 10000000000 100000000000 1000000000000; do
  for t in 1 12; do
    want=$($PS -t$t --no-status "$n" | awk '/Primes:/ {print $2}')
    got=$($BIN -t $t "$@" "$n" | awk '/^pi\(/ {print $3}')
    check "pi($n) t=$t" "$got" "$want"
  done
done

# boundary primes 786431^2 and 786433^2
check "pi(618473717761)" "$($BIN "$@" 618473717761 | awk '/^pi\(/ {print $3}')" "23688293324"
check "pi(618476863489)" "$($BIN "$@" 618476863489 | awk '/^pi\(/ {print $3}')" "23688409284"

# high-n pend-regression windows.  The pend-migration ordering bug made pi()
# wrong above ~2e12 (first-multiple offset can exceed one segment for
# p > ~1.1M); the pi() sweep above tops out at 1e12 and cannot see it.
# Each window costs ~0.1-1 s single-thread, so they are cheap to keep.
for w in "2000000000000 2000200000000" "3000000000000 3000200000000" \
         "5000000000000 5000200000000" "10000000000000 10000010000000" \
         "20000000000000 20002000000000"; do
  lo=${w% *}; hi=${w#* }
  want=$($PS --no-status "$lo" "$hi" | awk '/Primes:/ {print $2}')
  got=$($BIN "$@" --lo "$lo" "$hi" | awk '/^pi\(\[/ {print $4}')
  check "pi([$lo,$hi])" "$got" "$want"
done

# segment-edge interval windows: countLo/cap landing exactly on a segment
# boundary (or +/-1) takes the prefix-difference/phantom counting paths the
# full-range sweep never exercises.  span = 30*B = 7864320 (default B).
for w in "7864320 8864320" "7864319 8864319" "7864320 7864321" \
         "15728640 16728640" "15728639 16728639" "15728640 15728641" \
         "23592960 24592960" "23592959 24592959" "23592960 23592961" \
         "9174900 9274900" "7864320 15728640"; do
  lo=${w% *}; hi=${w#* }
  want=$($PS --no-status "$lo" "$hi" | awk '/Primes:/ {print $2}')
  got=$($BIN "$@" --lo "$lo" "$hi" | awk '/^pi\(\[/ {print $4}')
  check "pi([$lo,$hi])" "$got" "$want"
done

echo "----------------------------------------"
echo "pass=$pass fail=$fail"
[ $fail -eq 0 ]
