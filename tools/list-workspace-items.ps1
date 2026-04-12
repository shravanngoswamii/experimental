param(
  [ValidateSet("all", "node", "vite", "julia", "html", "astro-pages", "markdown", "manifests")]
  [string]$Kind = "all",
  [string]$Root = ".",
  [string]$AstroPagesRoot = "sample/src/pages",
  [switch]$AbsolutePath,
  [string[]]$Extensions = @("html"),
  [switch]$CountOnly
)

$rootPath = (Resolve-Path -Path $Root).Path
$excludePattern = "\.git|node_modules|dist|build|\.next|\.astro|\.vite|\.quarto"

function Convert-ToDisplayPath {
  param(
    [string]$Path,
    [string]$Base,
    [switch]$UseAbsolute
  )

  if ($UseAbsolute) {
    return $Path
  }

  return $Path.Replace($Base, ".").TrimStart('\\')
}

function List-NodeProjects {
  Get-ChildItem -Path $rootPath -Recurse -File -Filter "package.json" |
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
        Path = Convert-ToDisplayPath -Path $projectDir -Base $rootPath -UseAbsolute:$AbsolutePath
      }
    } |
    Sort-Object Path
}

function List-ViteProjects {
  Get-ChildItem -Path $rootPath -Recurse -File -Filter "vite.config.ts" |
    Where-Object { $_.FullName -notmatch $excludePattern } |
    ForEach-Object {
      $projectDir = $_.DirectoryName
      [PSCustomObject]@{
        Name = Split-Path $projectDir -Leaf
        Path = Convert-ToDisplayPath -Path $projectDir -Base $rootPath -UseAbsolute:$AbsolutePath
      }
    } |
    Sort-Object Path
}

function List-JuliaProjects {
  Get-ChildItem -Path $rootPath -Recurse -File -Filter "Project.toml" |
    Where-Object { $_.FullName -notmatch $excludePattern } |
    ForEach-Object {
      $projectDir = $_.DirectoryName
      [PSCustomObject]@{
        Name = Split-Path $projectDir -Leaf
        Path = Convert-ToDisplayPath -Path $projectDir -Base $rootPath -UseAbsolute:$AbsolutePath
      }
    } |
    Sort-Object Path
}

function List-HtmlTargets {
  $includePattern = @($Extensions | ForEach-Object { $_.Trim().TrimStart('.') }) -join '|'

  Get-ChildItem -Path $rootPath -Recurse -File |
    Where-Object { $_.Extension.TrimStart('.') -match "^($includePattern)$" } |
    Where-Object { $_.FullName -notmatch $excludePattern } |
    Sort-Object FullName |
    Select-Object @{Name = "Path"; Expression = { Convert-ToDisplayPath -Path $_.FullName -Base $rootPath -UseAbsolute:$AbsolutePath }}
}

function List-AstroPages {
  if (-not (Test-Path $AstroPagesRoot)) {
    Write-Error "Path not found: $AstroPagesRoot"
    return @()
  }

  $astroPath = (Resolve-Path -Path $AstroPagesRoot).Path

  Get-ChildItem -Path $astroPath -Recurse -File |
    Where-Object { $_.Extension -in @('.astro', '.md', '.mdx', '.html') } |
    Sort-Object FullName |
    Select-Object @{Name = "Path"; Expression = { Convert-ToDisplayPath -Path $_.FullName -Base $astroPath -UseAbsolute:$AbsolutePath }}
}

function List-MarkdownFiles {
  Get-ChildItem -Path $rootPath -Recurse -File |
    Where-Object { $_.Extension -in @('.md', '.qmd') } |
    Where-Object { $_.FullName -notmatch $excludePattern } |
    Sort-Object FullName |
    Select-Object @{Name = "Path"; Expression = { Convert-ToDisplayPath -Path $_.FullName -Base $rootPath -UseAbsolute:$AbsolutePath }}
}

function List-ProjectManifests {
  $manifestNames = @("package.json", "Project.toml", "pyproject.toml")

  Get-ChildItem -Path $rootPath -Recurse -File |
    Where-Object { $_.Name -in $manifestNames } |
    Where-Object { $_.FullName -notmatch $excludePattern } |
    Sort-Object FullName |
    Select-Object Name, @{Name = "Path"; Expression = { Convert-ToDisplayPath -Path $_.FullName -Base $rootPath -UseAbsolute:$AbsolutePath }}
}

$data = switch ($Kind) {
  "node" { List-NodeProjects }
  "vite" { List-ViteProjects }
  "julia" { List-JuliaProjects }
  "html" { List-HtmlTargets }
  "astro-pages" { List-AstroPages }
  "markdown" { List-MarkdownFiles }
  "manifests" { List-ProjectManifests }
  "all" {
    @(
      [PSCustomObject]@{ Kind = "node"; Count = (List-NodeProjects).Count }
      [PSCustomObject]@{ Kind = "vite"; Count = (List-ViteProjects).Count }
      [PSCustomObject]@{ Kind = "julia"; Count = (List-JuliaProjects).Count }
      [PSCustomObject]@{ Kind = "html"; Count = (List-HtmlTargets).Count }
      [PSCustomObject]@{ Kind = "astro-pages"; Count = (List-AstroPages).Count }
      [PSCustomObject]@{ Kind = "markdown"; Count = (List-MarkdownFiles).Count }
      [PSCustomObject]@{ Kind = "manifests"; Count = (List-ProjectManifests).Count }
    )
  }
}

if ($CountOnly -and $Kind -ne "all") {
  $data.Count
}
else {
  $data | Format-Table -AutoSize
}
