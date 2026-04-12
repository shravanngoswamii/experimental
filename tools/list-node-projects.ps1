param(
  [string]$Root = ".",
  [switch]$AbsolutePath
)

$excludePattern = "node_modules|dist|build|\.git|\.next|\.astro|\.vite"
$resolvedRoot = (Resolve-Path -Path $Root).Path

$projects = Get-ChildItem -Path $resolvedRoot -Recurse -File -Filter "package.json" |
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
      Path = if ($AbsolutePath) {
        $projectDir
      }
      else {
        $projectDir.Replace($resolvedRoot, ".").TrimStart('\\')
      }
    }
  } |
  Sort-Object Path

$projects |
  Format-Table -AutoSize
