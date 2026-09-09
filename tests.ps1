# tests.ps1 - fastsieve system test suite.
# Validates the CPU engine (single & multi thread) and the GPU accelerator
# against a hard-coded oracle of exact values of pi(n).
$ErrorActionPreference = 'Stop'
$tool = Join-Path $PSScriptRoot 'fastsieve.exe'
if (-not (Test-Path $tool)) { throw "build fastsieve.exe first" }

$cases = @(
  @{ n = 100;           e = [uint64]25 },
  @{ n = 1000;          e = [uint64]168 },
  @{ n = 10000;         e = [uint64]1229 },
  @{ n = 100000;        e = [uint64]9592 },
  @{ n = 1000000;       e = [uint64]78498 },
  @{ n = 10000000;      e = [uint64]664579 },
  @{ n = 100000000;     e = [uint64]5761455 },
  @{ n = 1000000000;    e = [uint64]50847534 },
  @{ n = 10000000000;   e = [uint64]455052511 },
  @{ n = 100000000000;  e = [uint64]4118054813 },
  @{ n = 1000000000000; e = [uint64]37607912018 },
  # trailing-boundary (value==k*30B+1 = composite 786431^2, c.f. 786433^2)
  @{ n = 618473717761;  e = [uint64]23688293324 },
  @{ n = 618476863489;  e = [uint64]23688409284 },
  # single 16KiB GPU block boundaries
  @{ n = 491520;        e = [uint64]40883 },
  @{ n = 983040;        e = [uint64]77279 }
)

function Run-Case($n, $argz) {
  if ($env:TESTDBG) { Write-Host ("DBG invoking: {0} {1} {2}" -f $tool, ($argz -join ' '), $n) }
  # Native stderr (e.g. GPU audit/fallback notices) must not terminate the
  # suite under $ErrorActionPreference='Stop'; relax it for the native call.
  $ErrorActionPreference = 'Continue'
  $o = & $tool @argz $n 2>$null | Select-String -Pattern '^pi\('
  if (-not $o) { return $null }
  return [uint64]([regex]::Match($o.Line, '^pi\(\d+\) = (\d+)').Groups[1].Value)
}

$fail = 0
foreach ($c in $cases) {
  $n = $c.n
  foreach ($mode in 't1','t12','gpu') {
    if ($mode -eq 't1')      { $argz = '-t','1' }
    elseif ($mode -eq 't12') { $argz = '-t','12' }
    else                     { $argz = ,'--gpu' }
    # big quadrat and GPU not needed for the largest ranges to keep the suite quick
    if ($mode -eq 'gpu' -and $n -gt 100000000000) { continue }
    $got = Run-Case $n $argz
    if ($null -eq $got) { Write-Host ("FAIL {0,-15} {1}  (no output / crashed)" -f $n, $mode) -ForegroundColor Red; $fail++; continue }
    if ($got -eq $c.e) { Write-Host ("PASS {0,-15} {1}  = {2}" -f $n, $mode, $got) -ForegroundColor Green }
    else { Write-Host ("FAIL {0,-15} {1}  got {2} want {3}" -f $n, $mode, $got, $c.e) -ForegroundColor Red; $fail++ }
  }
}
# Slice-window regressions (--lo/--hi, cost ~seconds).  Two families that the
# full-range pi sweep cannot see:
#   * pend migration bug (pi wrong above ~2e12; first-multiple offset can
#     exceed one segment for p > ~1.1M) -> windows at 2e12..2e13;
#   * countLo/cap landing exactly on a segment edge (phantom off-by-one)
#     -> the k*span windows below (span = 30*B = 7864320 at default B).
# Oracle values are primesieve-verified.
function Test-Win($c) {
  $o = & $tool --lo $c.lo $c.hi 2>$null | Select-String -Pattern '^pi\(\[' | Select-Object -First 1
  if (-not $o) { Write-Host ("FAIL win [{0},{1}]  (no output / crashed)" -f $c.lo, $c.hi) -ForegroundColor Red; $fail++; return }
  $g = [uint64]([regex]::Match($o.Line, '= (\d+)').Groups[1].Value)
  if ($g -eq $c.e) { Write-Host ("PASS win [{0},{1}]  = {2}" -f $c.lo, $c.hi, $g) -ForegroundColor Green }
  else { Write-Host ("FAIL win [{0},{1}]  got {2} want {3}" -f $c.lo, $c.hi, $g, $c.e) -ForegroundColor Red; $fail++ }
}
$winCases = @(
  # pend-regression (activation past ~1.3e12)
  @{ lo = 2000000000000;  hi = 2000200000000;   e = [uint64]7061729 },
  @{ lo = 3000000000000;  hi = 3000200000000;   e = [uint64]6962573 },
  @{ lo = 5000000000000;  hi = 5000200000000;   e = [uint64]6839252 },
  @{ lo = 10000000000000; hi = 10000010000000;  e = [uint64]334312 },
  @{ lo = 20000000000000; hi = 20002000000000;  e = [uint64]65307881 },
  # segment-edge counting (k*span and +/-1, plus the original repro)
  @{ lo = 7864320;   hi = 8864320;   e = [uint64]62736 },
  @{ lo = 7864319;   hi = 8864319;   e = [uint64]62736 },
  @{ lo = 7864320;   hi = 7864321;   e = [uint64]0 },
  @{ lo = 9174900;   hi = 9274900;   e = [uint64]6287 },
  @{ lo = 15728640;  hi = 16728640;  e = [uint64]60193 },
  @{ lo = 15728639;  hi = 16728639;  e = [uint64]60193 },
  @{ lo = 15728640;  hi = 15728641;  e = [uint64]0 },
  @{ lo = 23592960;  hi = 24592960;  e = [uint64]58875 },
  @{ lo = 23592959;  hi = 24592959;  e = [uint64]58875 },
  @{ lo = 23592960;  hi = 23592961;  e = [uint64]0 },
  @{ lo = 7864320;   hi = 15728640;  e = [uint64]483481 }
)
foreach ($c in $winCases) { Test-Win $c }
# differential vs reference primesieve if available
$ref = Get-Command primesieve -ErrorAction SilentlyContinue
if ($ref) {
  foreach ($n in 10000000, 100000000, 1000000000) {
    $r = (& $ref.Source -t1 --no-status $n 2>$null | Select-String -Pattern 'Primes:').Line -replace '.*Primes: ',''
    $g = Run-Case $n @('-t','1') 't1'
    if ($g -eq [uint64]$r) { Write-Host ("PASS ref-diff {0}" -f $n) -ForegroundColor Green }
    else { Write-Host ("FAIL ref-diff {0} ours={1} primesieve={2}" -f $n, $g, $r) -ForegroundColor Red; $fail++ }
  }
} else { Write-Host "note: tuples of primesieve not found; skip reference differential" -ForegroundColor DarkGray }

Write-Host ("RESULT: {0} failure(s)" -f $fail)
if ($fail -gt 0) { exit 1 } else { exit 0 }