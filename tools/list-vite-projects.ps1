param(
  [string]$Root = ".",
  [switch]$AbsolutePath
)

$excludePattern = "\.git|node_modules|dist|build"
$resolvedRoot = (Resolve-Path -Path $Root).Path

Get-ChildItem -Path $resolvedRoot -Recurse -File -Filter "vite.config.ts" |
  Where-Object { $_.FullName -notmatch $excludePattern } |
  ForEach-Object {
    $projectDir = $_.DirectoryName

    [PSCustomObject]@{
      Name = Split-Path $projectDir -Leaf
      Path = if ($AbsolutePath) {
        $projectDir
      }
      else {
        $projectDir.Replace($resolvedRoot, ".").TrimStart('\\')
      }
    }
  } |
  Sort-Object Path |
  Format-Table -AutoSize
