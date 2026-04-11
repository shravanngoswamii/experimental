param(
  [string]$Root = ".",
  [switch]$AbsolutePath
)

$excludePattern = "node_modules|dist|build|\.git|\.astro|\.vite"

$resolvedRoot = (Resolve-Path -Path $Root).Path

$results = Get-ChildItem -Path $resolvedRoot -Recurse -File -Filter "*.html" |
  Where-Object { $_.FullName -notmatch $excludePattern } |
  Sort-Object FullName

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
