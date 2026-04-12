param(
  [string]$Root = ".",
  [switch]$AbsolutePath,
  [string[]]$Extensions = @("html"),
  [switch]$CountOnly
)

$excludePattern = "node_modules|dist|build|\.git|\.astro|\.vite"

$resolvedRoot = (Resolve-Path -Path $Root).Path

$includePattern = @($Extensions | ForEach-Object { $_.Trim().TrimStart('.') }) -join '|'

$results = Get-ChildItem -Path $resolvedRoot -Recurse -File |
  Where-Object { $_.Extension.TrimStart('.') -match "^($includePattern)$" } |
  Where-Object { $_.FullName -notmatch $excludePattern } |
  Sort-Object FullName

if ($CountOnly) {
  $results.Count
  return
}

if ($AbsolutePath) {
  $results |
    Select-Object @{Name = "Path"; Expression = { $_.FullName }} |
    Format-Table -AutoSize
}
else {
  $results |
    Select-Object @{Name = "Path"; Expression = { $_.FullName.Replace($resolvedRoot, ".").TrimStart('\\') }} |
    Format-Table -AutoSize
}
