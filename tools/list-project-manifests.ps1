param(
  [string]$Root = ".",
  [switch]$AbsolutePath
)

$excludePattern = "\.git|node_modules|dist|build|\.astro|\.vite"
$resolvedRoot = (Resolve-Path -Path $Root).Path
$manifestNames = @("package.json", "Project.toml", "pyproject.toml")

Get-ChildItem -Path $resolvedRoot -Recurse -File |
  Where-Object { $_.Name -in $manifestNames } |
  Where-Object { $_.FullName -notmatch $excludePattern } |
  Sort-Object FullName |
  Select-Object Name, @{
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
