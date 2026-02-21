/*
 * Stub implementations for GAC functions not needed on Wine.
 * Signatures must match original gacutil.cpp exactly for P/Invoke compatibility.
 */

#include "EasyHookDll/stdafx.h"
#include <objbase.h>

/* ============ GAC stubs (matching original signatures) ============ */

/* Original: LPINTERNAL_CONTEXT __stdcall GacCreateContext() → @0 */
EXTERN_C __declspec(dllexport) void* __stdcall GacCreateContext(void)
{
    return NULL;
}

/* Original: BOOL __stdcall GacInstallAssembly(ctx, path, desc, uniqueID) → @16 */
EXTERN_C __declspec(dllexport) BOOL __stdcall GacInstallAssembly(
    void* InContext, WCHAR* InAssemblyPath, WCHAR* InDescription, WCHAR* InUniqueID)
{
    return FALSE;
}

/* Original: BOOL __stdcall GacUninstallAssembly(ctx, name, desc, uniqueID) → @16 */
EXTERN_C __declspec(dllexport) BOOL __stdcall GacUninstallAssembly(
    void* InContext, WCHAR* InAssemblyName, WCHAR* InDescription, WCHAR* InUniqueID)
{
    return FALSE;
}

/* Original: void __stdcall GacReleaseContext(LPINTERNAL_CONTEXT*) → @4 */
EXTERN_C __declspec(dllexport) void __stdcall GacReleaseContext(void** RefContext)
{
    if (RefContext) *RefContext = NULL;
}
