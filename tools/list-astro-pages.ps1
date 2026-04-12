param(
  [string]$Root = "sample/src/pages",
  [switch]$AbsolutePath
)

if (-not (Test-Path $Root)) {
  Write-Host "Path not found: $Root"
  exit 1
}

$resolvedRoot = (Resolve-Path -Path $Root).Path

Get-ChildItem -Path $resolvedRoot -Recurse -File |
  Where-Object { $_.Extension -in @('.astro', '.md', '.mdx', '.html') } |
  Sort-Object FullName |
  Select-Object @{
    Name = "Path"
    Expression = {
      if ($AbsolutePath) {
        $_.FullName
      }
      else {
        $_.FullName.Replace($resolvedRoot, ".").TrimStart('\\')
      }
    }
  } |
  Format-Table -AutoSize
