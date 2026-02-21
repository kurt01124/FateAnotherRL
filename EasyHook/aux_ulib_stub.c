/*
 * AuxUlib Wine-compatible implementation
 * Replaces Microsoft's aux_ulib.lib which is not available on Wine.
 *
 * Instead of always returning FALSE (no loader lock), this properly checks
 * the PEB's LoaderLock critical section to detect if the current thread
 * holds the OS loader lock. This prevents hooks from executing during
 * DLL loading, avoiding deadlocks in managed injection.
 *
 * PEB layout (x86, 32-bit):
 *   PEB->LoaderLock at offset 0xA0 = PRTL_CRITICAL_SECTION
 *   CRITICAL_SECTION->OwningThread at offset 0x0C
 */

#include <windows.h>
#include <winternl.h>

/* PEB->LoaderLock offset for 32-bit Windows */
#define PEB_LOADER_LOCK_OFFSET 0xA0

/* RTL_CRITICAL_SECTION->OwningThread offset */
#define CS_OWNING_THREAD_OFFSET 0x0C

static BOOL g_Initialized = FALSE;

BOOL WINAPI AuxUlibInitialize(void)
{
    g_Initialized = TRUE;
    return TRUE;
}

BOOL WINAPI AuxUlibIsDLLSynchronizationHeld(PBOOL SynchronizationHeld)
{
    PPEB pPeb;
    PRTL_CRITICAL_SECTION pLoaderLock;
    HANDLE currentThreadId;

    if (SynchronizationHeld == NULL)
        return FALSE;

    if (!g_Initialized)
    {
        SetLastError(ERROR_INVALID_FUNCTION);
        return FALSE;
    }

    /* Get PEB from TEB (fs:[0x30] on x86) */
#if defined(_M_IX86) || defined(__i386__)
    __asm {
        mov eax, dword ptr fs:[0x30]
        mov pPeb, eax
    }
#else
    /* Fallback: use NtCurrentTeb() */
    pPeb = NtCurrentTeb()->ProcessEnvironmentBlock;
#endif

    if (pPeb == NULL)
    {
        *SynchronizationHeld = FALSE;
        return TRUE;
    }

    /* PEB->LoaderLock is at offset 0xA0 for 32-bit */
    pLoaderLock = *(PRTL_CRITICAL_SECTION*)((BYTE*)pPeb + PEB_LOADER_LOCK_OFFSET);

    if (pLoaderLock == NULL)
    {
        *SynchronizationHeld = FALSE;
        return TRUE;
    }

    /* Check if current thread owns the loader lock */
    currentThreadId = (HANDLE)(ULONG_PTR)GetCurrentThreadId();

    if (pLoaderLock->OwningThread == currentThreadId && pLoaderLock->RecursionCount > 0)
    {
        *SynchronizationHeld = TRUE;
    }
    else
    {
        *SynchronizationHeld = FALSE;
    }

    return TRUE;
}
