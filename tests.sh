#!/bin/bash
# Verification sweep per docs/GPU_DEVELOPMENT.md section 4.
# Usage: ./tests.sh [binary] [extra args, e.g. --gpu]
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
         "10000000000000 10000010000000" "20000000000000 20002000000000"; do
  lo=${w% *}; hi=${w#* }
  want=$($PS --no-status "$lo" "$hi" | awk '/Primes:/ {print $2}')
  got=$($BIN "$@" --lo "$lo" "$hi" | awk '/^pi\(\[/ {print $4}')
  check "pi([$lo,$hi])" "$got" "$want"
done

echo "----------------------------------------"
echo "pass=$pass fail=$fail"
[ $fail -eq 0 ]
