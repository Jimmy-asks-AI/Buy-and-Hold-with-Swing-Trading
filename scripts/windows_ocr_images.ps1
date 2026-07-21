param(
    [Parameter(Mandatory = $true)]
    [string]$InputDirectory,

    [Parameter(Mandatory = $true)]
    [string]$OutputPath,

    [string]$LanguageTag = "zh-Hans-CN"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Storage.StorageFile, Windows.Storage, ContentType = WindowsRuntime]
$null = [Windows.Storage.FileAccessMode, Windows.Storage, ContentType = WindowsRuntime]
$null = [Windows.Storage.Streams.IRandomAccessStream, Windows.Storage.Streams, ContentType = WindowsRuntime]
$null = [Windows.Graphics.Imaging.BitmapDecoder, Windows.Foundation, ContentType = WindowsRuntime]
$null = [Windows.Graphics.Imaging.SoftwareBitmap, Windows.Foundation, ContentType = WindowsRuntime]
$null = [Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType = WindowsRuntime]
$null = [Windows.Media.Ocr.OcrResult, Windows.Foundation, ContentType = WindowsRuntime]
$null = [Windows.Globalization.Language, Windows.Foundation, ContentType = WindowsRuntime]

function Wait-WinRtOperation {
    param(
        [Parameter(Mandatory = $true)]$Operation,
        [Parameter(Mandatory = $true)][Type]$ResultType
    )

    $method = [System.WindowsRuntimeSystemExtensions].GetMethods() |
        Where-Object {
            $_.Name -eq "AsTask" -and
            $_.IsGenericMethod -and
            $_.GetParameters().Count -eq 1
        } |
        Select-Object -First 1
    if ($null -eq $method) {
        throw "Windows Runtime AsTask adapter is unavailable."
    }
    $task = $method.MakeGenericMethod($ResultType).Invoke($null, @($Operation))
    $task.Wait()
    return $task.Result
}

$resolvedInput = (Resolve-Path -LiteralPath $InputDirectory).Path
$images = @(Get-ChildItem -LiteralPath $resolvedInput -Filter "*.png" -File | Sort-Object Name)
if ($images.Count -eq 0) {
    throw "No PNG pages found in $resolvedInput"
}

$available = [Windows.Media.Ocr.OcrEngine]::AvailableRecognizerLanguages |
    Where-Object LanguageTag -eq $LanguageTag
if ($null -eq $available) {
    throw "Windows OCR language is unavailable: $LanguageTag"
}
$language = [Windows.Globalization.Language]::new($LanguageTag)
$engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage($language)
if ($null -eq $engine) {
    throw "Windows OCR engine could not be created for $LanguageTag"
}

$pages = [System.Collections.Generic.List[string]]::new()
$pageNumber = 0
foreach ($image in $images) {
    $pageNumber += 1
    $file = Wait-WinRtOperation ([Windows.Storage.StorageFile]::GetFileFromPathAsync($image.FullName)) ([Windows.Storage.StorageFile])
    $stream = Wait-WinRtOperation ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
    try {
        $decoder = Wait-WinRtOperation ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
        $bitmap = Wait-WinRtOperation ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
        try {
            $result = Wait-WinRtOperation ($engine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])
            $pages.Add("=== PAGE $pageNumber ===`n$($result.Text)")
        }
        finally {
            if ($bitmap -is [System.IDisposable]) {
                $bitmap.Dispose()
            }
        }
    }
    finally {
        if ($stream -is [System.IDisposable]) {
            $stream.Dispose()
        }
    }
}

$parent = Split-Path -Parent $OutputPath
if ($parent) {
    [System.IO.Directory]::CreateDirectory($parent) | Out-Null
}
$utf8 = [System.Text.UTF8Encoding]::new($false)
[System.IO.File]::WriteAllText($OutputPath, ($pages -join "`n`n"), $utf8)
