# Run in a normal PowerShell terminal, outside the restricted Codex session.
$ErrorActionPreference = 'Stop'

if (-not [string]::IsNullOrWhiteSpace($PSScriptRoot)) {
    $projectRoot = Split-Path -Parent $PSScriptRoot
} else {
    # PSScriptRoot is empty when the script is pasted into an interactive shell.
    $projectRoot = (Get-Location).Path
}
if (-not (Test-Path -LiteralPath (Join-Path $projectRoot 'PRD.md') -PathType Leaf)) {
    throw 'Root proyek tidak ditemukan. Jalankan file skrip, atau tempel isinya dari direktori E:\gta-vi-reviews.'
}
$draftPath = Join-Path $projectRoot 'skill-drafts\gta-vi-sentiment-analysis\SKILL.md'
$skillDirectory = Join-Path $projectRoot '.agents\skills\gta-vi-sentiment-analysis'
$destinationPath = Join-Path $skillDirectory 'SKILL.md'
$reportPath = Join-Path $projectRoot 'reports\skill_validation.json'

if (-not (Test-Path -LiteralPath $draftPath -PathType Leaf)) {
    throw "Draft skill tidak ditemukan: $draftPath"
}

$draftText = Get-Content -LiteralPath $draftPath -Raw -Encoding UTF8
if ($draftText -notmatch '(?ms)\A---\r?\nname: gta-vi-sentiment-analysis\r?\ndescription: [^\r\n]+\r?\n---\r?\n') {
    throw 'Frontmatter draft skill tidak sesuai; tidak ada file yang disalin.'
}
$sourceHash = (Get-FileHash -LiteralPath $draftPath -Algorithm SHA256).Hash

if (Test-Path -LiteralPath $destinationPath) {
    if (-not (Test-Path -LiteralPath $destinationPath -PathType Leaf)) {
        throw "Tujuan bukan file: $destinationPath"
    }
    if ((Get-FileHash -LiteralPath $destinationPath -Algorithm SHA256).Hash -ne $sourceHash) {
        throw 'Skill tujuan sudah ada dengan isi berbeda. File tersebut tidak ditimpa.'
    }
} else {
    if (-not (Test-Path -LiteralPath $skillDirectory -PathType Container)) {
        New-Item -Path $skillDirectory -ItemType Directory -Force | Out-Null
    }
    # No -Force: preserve a destination created by another process.
    Copy-Item -LiteralPath $draftPath -Destination $destinationPath
}

if ((Get-FileHash -LiteralPath $destinationPath -Algorithm SHA256).Hash -ne $sourceHash) {
    throw 'Checksum skill tujuan berbeda dari draft.'
}

if (Test-Path -LiteralPath $reportPath -PathType Leaf) {
    $installationReport = Get-Content -LiteralPath $reportPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $installationReport.installed_at_requested_path = $true
    $installationReport.placement_blocker = $null
    $installationReport.sha256 = $sourceHash.ToLowerInvariant()
    $installationReport | Add-Member -NotePropertyName 'installed_utc' -NotePropertyValue ([DateTime]::UtcNow.ToString('o')) -Force
    $installationReport | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $reportPath -Encoding UTF8
}

$readmePath = Join-Path $projectRoot 'README.md'
if (Test-Path -LiteralPath $readmePath -PathType Leaf) {
    $readmeText = (Get-Content -LiteralPath $readmePath -Raw -Encoding UTF8).Replace("`r`n", "`n")
    $previousParagraph = @'
Skill proyek telah disiapkan sebagai draft pada
`skill-drafts/gta-vi-sentiment-analysis/SKILL.md`. Penempatan yang diminta ke
`.agents/skills/gta-vi-sentiment-analysis/` terhalang izin read-only sesi ini;
draft belum terpasang sebagai skill lokal di direktori tersebut.
'@
    $installedParagraph = @'
Skill proyek tersedia di `.agents/skills/gta-vi-sentiment-analysis/SKILL.md`.
Salinan sumber tetap tersedia di `skill-drafts/gta-vi-sentiment-analysis/SKILL.md`.
Checksum pemasangan dicatat di `reports/skill_validation.json`.
'@
    $updatedReadme = $readmeText.Replace($previousParagraph.Replace("`r`n", "`n"), $installedParagraph.Replace("`r`n", "`n"))
    if ($updatedReadme -ne $readmeText) {
        [System.IO.File]::WriteAllText($readmePath, $updatedReadme, [System.Text.UTF8Encoding]::new($false))
    }
}

Write-Output "Skill terpasang dan checksum cocok: $destinationPath"
Write-Output 'Dataset, split, TF-IDF, model, dan hasil evaluasi tidak diubah.'
