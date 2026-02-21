$ErrorActionPreference = "Stop"

# VS2019 vcvarsall via Invoke-BatchFile (from build.ps1 pattern)
function Invoke-BatchFile {
    param([string]$Path, [string]$Parameters)
    $tempFile = [IO.Path]::GetTempFileName()
    cmd.exe /c " `"$Path`" $Parameters && set > `"$tempFile`" "
    Get-Content $tempFile | Foreach-Object {
        if ($_ -match "^(.*?)=(.*)$") {
            Set-Content "env:\$($matches[1])" $matches[2]
        }
    }
    Remove-Item $tempFile
}

$vcvars = 'C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat'
if (!(Test-Path $vcvars)) { throw "VS2019 vcvarsall.bat not found" }
Write-Host "Loading VS2019 x86 environment..."
Invoke-BatchFile $vcvars "x86"

Set-Location 'C:\Users\kurtz\Desktop\FateAnother\src\EasyHook'
if (!(Test-Path build)) { New-Item -ItemType Directory -Path build | Out-Null }

$INC = '/I"Public" /I"EasyHookDll" /I"DriverShared" /I"DriverShared\Disassembler" /I"DriverShared\Disassembler\libudis86"'
$COMMON = "/nologo /c /Ox /Oi /Ot /GF /Gz /GS- /EHa /GL /MT /W3 /DEASYHOOK_EXPORTS /DWIN32 /D_WINDOWS /DUNICODE /D_UNICODE $INC"
$CFLAGS = "$COMMON /TP"
$CFLAGS_C = "$COMMON /TC"

Write-Host "[1/10] ASM..."
& ml /nologo /c /Fo build\HookSpecific_x86.obj "DriverShared\ASM\HookSpecific_x86.asm"
if ($LASTEXITCODE -ne 0) { throw "FAIL: MASM" }

Write-Host "[2/10] AuxUlib PEB stub..."
Invoke-Expression "cl $CFLAGS_C /Fo""build\aux_ulib_stub.obj"" aux_ulib_stub.c"
if ($LASTEXITCODE -ne 0) { throw "FAIL: aux_ulib_stub" }

Write-Host "[3/10] Core..."
Invoke-Expression "cl $CFLAGS /Fo""build\dllmain.obj"" EasyHookDll\dllmain.c"
Invoke-Expression "cl $CFLAGS /Fo""build\acl.obj"" EasyHookDll\LocalHook\acl.c"
Invoke-Expression "cl $CFLAGS /Fo""build\memory.obj"" EasyHookDll\Rtl\memory.c"
Invoke-Expression "cl $CFLAGS /Fo""build\file.obj"" EasyHookDll\Rtl\file.c"
if ($LASTEXITCODE -ne 0) { throw "FAIL: core" }

Write-Host "[4/10] RTL..."
Invoke-Expression "cl $CFLAGS /Fo""build\error.obj"" DriverShared\Rtl\error.c"
Invoke-Expression "cl $CFLAGS /Fo""build\string.obj"" DriverShared\Rtl\string.c"
if ($LASTEXITCODE -ne 0) { throw "FAIL: rtl" }

Write-Host "[5/10] LocalHook..."
Invoke-Expression "cl $CFLAGS /Fo""build\barrier.obj"" DriverShared\LocalHook\barrier.c"
Invoke-Expression "cl $CFLAGS /Fo""build\install.obj"" DriverShared\LocalHook\install.c"
Invoke-Expression "cl $CFLAGS /Fo""build\uninstall.obj"" DriverShared\LocalHook\uninstall.c"
Invoke-Expression "cl $CFLAGS /Fo""build\reloc.obj"" DriverShared\LocalHook\reloc.c"
Invoke-Expression "cl $CFLAGS /Fo""build\alloc.obj"" DriverShared\LocalHook\alloc.c"
Invoke-Expression "cl $CFLAGS /Fo""build\caller.obj"" DriverShared\LocalHook\caller.c"
if ($LASTEXITCODE -ne 0) { throw "FAIL: localhook" }

Write-Host "[6/10] udis86..."
Invoke-Expression "cl $CFLAGS_C /Fo""build\udis86.obj"" DriverShared\Disassembler\libudis86\udis86.c"
Invoke-Expression "cl $CFLAGS_C /Fo""build\decode.obj"" DriverShared\Disassembler\libudis86\decode.c"
Invoke-Expression "cl $CFLAGS_C /Fo""build\itab.obj"" DriverShared\Disassembler\libudis86\itab.c"
Invoke-Expression "cl $CFLAGS_C /Fo""build\syn.obj"" DriverShared\Disassembler\libudis86\syn.c"
Invoke-Expression "cl $CFLAGS_C /Fo""build\syn-att.obj"" DriverShared\Disassembler\libudis86\syn-att.c"
Invoke-Expression "cl $CFLAGS_C /Fo""build\syn-intel.obj"" DriverShared\Disassembler\libudis86\syn-intel.c"
if ($LASTEXITCODE -ne 0) { throw "FAIL: udis86" }

Write-Host "[7/10] debug.cpp..."
Invoke-Expression "cl $CFLAGS /Fo""build\debug.obj"" EasyHookDll\LocalHook\debug.cpp"
if ($LASTEXITCODE -ne 0) { throw "FAIL: debug" }

Write-Host "[8/10] RemoteHook..."
Invoke-Expression "cl $CFLAGS /Fo""build\thread.obj"" EasyHookDll\RemoteHook\thread.c"
Invoke-Expression "cl $CFLAGS /Fo""build\stealth.obj"" EasyHookDll\RemoteHook\stealth.c"
Invoke-Expression "cl $CFLAGS /Fo""build\service.obj"" EasyHookDll\RemoteHook\service.c"
Invoke-Expression "cl $CFLAGS /Fo""build\entry.obj"" EasyHookDll\RemoteHook\entry.cpp"
Invoke-Expression "cl $CFLAGS /Fo""build\driver.obj"" EasyHookDll\RemoteHook\driver.cpp"
if ($LASTEXITCODE -ne 0) { throw "FAIL: remotehook" }

Write-Host "[9/10] GAC stubs..."
Invoke-Expression "cl $CFLAGS_C /Fo""build\stubs.obj"" stubs.c"
if ($LASTEXITCODE -ne 0) { throw "FAIL: stubs" }

Write-Host "[10/10] Linking..."
Set-Location build
& link /nologo /DLL /LTCG /SAFESEH:NO /OPT:REF /OPT:ICF /OUT:EasyHook32.dll `
    dllmain.obj acl.obj memory.obj file.obj `
    error.obj string.obj `
    barrier.obj install.obj uninstall.obj reloc.obj alloc.obj caller.obj `
    udis86.obj decode.obj itab.obj syn.obj syn-att.obj syn-intel.obj `
    debug.obj `
    thread.obj stealth.obj service.obj entry.obj driver.obj `
    stubs.obj aux_ulib_stub.obj HookSpecific_x86.obj `
    kernel32.lib user32.lib advapi32.lib ole32.lib psapi.lib shlwapi.lib mscoree.lib
if ($LASTEXITCODE -ne 0) { throw "FAIL: LINK" }

Write-Host ""
Write-Host "=== BUILD SUCCESSFUL ==="
Get-Item EasyHook32.dll | Select-Object Name, Length
