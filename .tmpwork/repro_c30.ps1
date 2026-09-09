$exe = "C:\Users\14255\AppData\Local\Programs\GAIA\gaia-agent.exe"
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = $exe
$psi.RedirectStandardInput = $true
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true
$psi.UseShellExecute = $false
$p = [System.Diagnostics.Process]::Start($psi)
Write-Output "parent pid = $($p.Id)"
for ($i=0; $i -lt 20; $i++) {
  Start-Sleep -Milliseconds 500
  $kids = @(Get-CimInstance Win32_Process -Filter "ParentProcessId = $($p.Id)")
  if ($kids.Count -gt 0) { break }
  if ($p.HasExited) { break }
}
Write-Output "parent exited early: $($p.HasExited)"
if ($p.HasExited) { Write-Output "stderr: $($p.StandardError.ReadToEnd())"; exit }
Write-Output "children: $(($kids | ForEach-Object { "$($_.ProcessId)/$($_.Name)" }) -join ', ')"
$p.Kill()
Start-Sleep -Seconds 2
foreach ($k in $kids) {
  $still = @(Get-CimInstance Win32_Process -Filter "ProcessId = $($k.ProcessId)")
  Write-Output "child $($k.ProcessId) alive after parent Kill(): $($still.Count -gt 0)"
  if ($still.Count -gt 0) { Stop-Process -Id $k.ProcessId -Force -ErrorAction SilentlyContinue }
}
