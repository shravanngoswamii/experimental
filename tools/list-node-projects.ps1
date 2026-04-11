param(
  [string]$Root = "."
)

Get-ChildItem -Path $Root -Recurse -File -Filter "package.json" |
  ForEach-Object {
    [PSCustomObject]@{
      Name = Split-Path $_.DirectoryName -Leaf
      Path = $_.DirectoryName
    }
  } |
  Sort-Object Path |
  Format-Table -AutoSize
