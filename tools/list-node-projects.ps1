param(
  [string]$Root = "."
)

$excludePattern = "node_modules|dist|build|\.git"

Get-ChildItem -Path $Root -Recurse -File -Filter "package.json" |
  Where-Object { $_.FullName -notmatch $excludePattern } |
  ForEach-Object {
    $projectDir = $_.DirectoryName
    $isVite = Test-Path (Join-Path $projectDir "vite.config.ts")
    $isAstro = Test-Path (Join-Path $projectDir "astro.config.mjs")

    $framework = if ($isAstro) {
      "Astro"
    }
    elseif ($isVite) {
      "Vite"
    }
    else {
      "Node"
    }

    [PSCustomObject]@{
      Name = Split-Path $projectDir -Leaf
      Framework = $framework
      Path = $projectDir
    }
  } |
  Sort-Object Path |
  Format-Table -AutoSize
