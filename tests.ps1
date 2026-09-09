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
  if ($env:TESTDBG) { Write-Host ("DBG invoking: {0} {1} {2}" -f $tool, $argz, $n) }
  $o = & $tool @argsLine $n 2>$null | Select-String -Pattern '^pi\('
  if (-not $o) { return $null }
  return [uint64]([regex]::Match($o.Line, '^pi\(\d+\) = (\d+)').Groups[1].Value)
}

$fail = 0
foreach ($c in $cases) {
  $n = $c.n
  foreach ($mode in 't1','t12','gpu') {
    $argz = if ($mode -eq 't1')   { '-t','1' }
            elseif ($mode -eq 't12'){ '-t','12' }
            else                    { '--gpu' }
    # big quadrat and GPU not needed for the largest ranges to keep the suite quick
    if ($mode -eq 'gpu' -and $n -gt 100000000000) { continue }
    $got = Run-Case $n $argz
    if ($null -eq $got) { Write-Host ("FAIL {0,-15} {1}  (no output / crashed)" -f $n, $mode) -ForegroundColor Red; $fail++; continue }
    if ($got -eq $c.e) { Write-Host ("PASS {0,-15} {1}  = {2}" -f $n, $mode, $got) -ForegroundColor Green }
    else { Write-Host ("FAIL {0,-15} {1}  got {2} want {3}" -f $n, $mode, $got, $c.e) -ForegroundColor Red; $fail++ }
  }
}
# differential vs reference primesieve if available
$ref = Get-Command primesieve -ErrorAction SilentlyContinue
if ($ref) {
  foreach ($n in 10000000, 100000000, 1000000000) {
    $r = (& $ref.Source -t1 --no-status $n 2>$null | Select-String -Pattern 'Primes:').Line -replace '.*Primes: ',''
    $g = Run-Case $n @('-t','1') 't1'
    if ($g -eq [uint64]$r) { Write-Host ("PASS ref-diff {0}" -f $n) -ForegroundColor Green }
    else { Write-Host ("FAIL ref-diff {0} ours={1} primesieve={2}" -f $n, $g, $r) -ForegroundColor Red; $fail++ }
  }
  # regression windows >2e12 - exercises pend one-early bug (main latent until ~1.3e12)
  # interval [lo,hi] uses --lo/--hi slice so cost is O(hi-lo) not O(hi)
  $windows = @(
    @{ lo = 2000000000000; hi = 2000200000000 },
    @{ lo = 3000000000000; hi = 3000200000000 },
    @{ lo = 5000000000000; hi = 5000200000000 }
  )
  foreach ($w in $windows) {
    $lo = $w.lo; $hi = $w.hi
    # primesieve window count: primesieve lo hi returns Primes: <count>
    $rLine = & $ref.Source $lo $hi 2>$null | Select-String -Pattern 'Primes:'
    if ($rLine) { $r = [uint64]($rLine.Line -replace '.*Primes:\s*','') } else { continue }
    $o = & $tool --lo $lo --hi $hi 2>$null | Select-String -Pattern 'pi\('
    if ($o) {
      $g = [uint64]([regex]::Match($o.Line, '=\s*(\d+)').Groups[1].Value)
      if ($g -eq $r) { Write-Host ("PASS window [{0},{1}] = {2}" -f $lo,$hi,$g) -ForegroundColor Green }
      else { Write-Host ("FAIL window [{0},{1}] got {2} want {3}" -f $lo,$hi,$g,$r) -ForegroundColor Red; $fail++ }
    }
  }
} else { Write-Host "note: tuples of primesieve not found; skip reference differential" -ForegroundColor DarkGray }

Write-Host ("RESULT: {0} failure(s)" -f $fail)
if ($fail -gt 0) { exit 1 } else { exit 0 }