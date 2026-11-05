# Render a deck's slides to PNG with PowerPoint, leaving any presentation the user has open alone.
param([string]$Deck, [string]$OutDir, [int[]]$Only)
$Deck = (Resolve-Path $Deck).Path
New-Item -ItemType Directory -Force $OutDir | Out-Null
$OutDir = (Resolve-Path $OutDir).Path
$app = New-Object -ComObject PowerPoint.Application
$already = $app.Presentations.Count
$pres = $app.Presentations.Open($Deck, -1, 0, 0)
try {
    foreach ($slide in $pres.Slides) {
        if ($Only -and ($Only -notcontains $slide.SlideIndex)) { continue }
        $slide.Export((Join-Path $OutDir ("slide-{0:d2}.png" -f $slide.SlideIndex)), "PNG", 1600, 900)
    }
    "rendered $($pres.Slides.Count) slides"
} finally {
    $pres.Close()
    if ($already -eq 0) { $app.Quit() }
}
