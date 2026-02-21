"""
FateAnother RL - Build & Assemble

EasyHook uses pre-built DLLs from thirdparty/ (source build has DllExport issues).
Build order: thirdparty -> JassNative -> JNLoader -> RLCommPlugin

Usage:
    python assemble.py              # Full build + War3Client assembly
    python assemble.py --skip-map   # Skip map patching
    python assemble.py --clean      # Clean rebuild
    python assemble.py --build-only # Build to out/ without War3Client assembly
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
import zipfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "out")
WAR3_ZIP = os.path.join(SCRIPT_DIR, "War3Client.zip")
WAR3_DIR = os.path.join(SCRIPT_DIR, "War3Client")
THIRDPARTY_DIR = os.path.join(SCRIPT_DIR, "thirdparty")

JASSNATIVE_DIR = os.path.join(SCRIPT_DIR, "JassNative")
JNLOADER_DIR = os.path.join(SCRIPT_DIR, "JNLoader")
RLCOMM_DIR = os.path.join(SCRIPT_DIR, "rl_comm")
MAPPATCH_DIR = os.path.join(SCRIPT_DIR, "MapPatch")


def log(msg, color=None):
    colors = {"green": "\033[92m", "yellow": "\033[93m", "red": "\033[91m", "cyan": "\033[96m"}
    reset = "\033[0m"
    if color and color in colors:
        print(f"{colors[color]}{msg}{reset}")
    else:
        print(msg)


def find_msbuild():
    candidates = []
    for year, base in [("2019", r"C:\Program Files (x86)"), ("2022", r"C:\Program Files")]:
        for edition in ["Community", "Professional", "Enterprise", "BuildTools"]:
            candidates.append(os.path.join(base, "Microsoft Visual Studio", year, edition,
                                           "MSBuild", "Current", "Bin", "MSBuild.exe"))
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def find_csc():
    candidates = []
    for year, base in [("2022", r"C:\Program Files"), ("2019", r"C:\Program Files (x86)")]:
        for edition in ["Community", "Professional", "Enterprise", "BuildTools"]:
            candidates.append(os.path.join(base, "Microsoft Visual Studio", year, edition,
                                           "MSBuild", "Current", "Bin", "Roslyn", "csc.exe"))
    candidates.append(r"C:\Windows\Microsoft.NET\Framework\v4.0.30319\csc.exe")
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def run_build(cmd, label):
    log(f"    {label}...", "yellow")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log(f"  FAILED: {label}", "red")
        output = (result.stdout + result.stderr).strip().split("\n")
        for line in output[-20:]:
            print(f"    {line}")
        sys.exit(1)


def copy_to_out(src, name=None):
    """Copy a file to out/"""
    dst = os.path.join(OUT_DIR, name or os.path.basename(src))
    if os.path.isfile(src):
        shutil.copy2(src, dst)
        return True
    return False


def require_in_out(files):
    """Verify files exist in out/"""
    missing = [f for f in files if not os.path.isfile(os.path.join(OUT_DIR, f))]
    if missing:
        log(f"  FAILED: Missing in out/: {missing}", "red")
        sys.exit(1)


# ============================================================
# Build Steps (all output to out/)
# ============================================================

def step_thirdparty():
    """Copy all thirdparty DLLs to out/ (includes pre-built EasyHook)"""
    if not os.path.isdir(THIRDPARTY_DIR):
        log("  WARNING: thirdparty/ not found", "yellow")
        return
    count = 0
    for f in os.listdir(THIRDPARTY_DIR):
        if f.endswith(".dll"):
            copy_to_out(os.path.join(THIRDPARTY_DIR, f))
            count += 1
    require_in_out(["EasyHook.dll", "EasyHook32.dll", "EasyLoad32.dll"])
    log(f"  {count} DLLs -> out/ (includes EasyHook pre-built)", "green")


def step_build_jassnative(msbuild):
    """Build JassNative -> out/Cirnix.JassNative.*.dll"""
    sln = os.path.join(JASSNATIVE_DIR, "JassNative.sln")
    if not os.path.isfile(sln):
        log("  WARNING: JassNative.sln not found", "yellow")
        return

    run_build([
        msbuild, sln,
        "/p:Configuration=Release", "/p:Platform=x86",
        "/t:Build", "/m", "/v:minimal",
    ], "JassNative.sln (Release|x86)")

    # Copy from JassNative build output
    jns = os.path.join(JASSNATIVE_DIR, "Build", "Release", "JNService")
    copy_to_out(os.path.join(jns, "Cirnix.JassNative.Runtime.dll"))
    for dll in ["Cirnix.JassNative.dll", "Cirnix.JassNative.Common.dll", "Cirnix.JassNative.YDWE.dll"]:
        copy_to_out(os.path.join(jns, "Plugins", dll))

    require_in_out(["Cirnix.JassNative.Runtime.dll", "Cirnix.JassNative.dll"])
    log("  JassNative -> out/", "green")


def step_build_jnloader(msbuild):
    """Build JNLoader -> out/JNLoader.exe"""
    sln = os.path.join(JNLOADER_DIR, "JNLoader.sln")
    if not os.path.isfile(sln):
        log("  WARNING: JNLoader.sln not found", "yellow")
        return

    run_build([
        msbuild, sln,
        "/p:Configuration=Release", "/p:Platform=x86",
        "/t:Build", "/m", "/v:minimal",
    ], "JNLoader.sln (Release|x86)")

    build_dir = os.path.join(JNLOADER_DIR, "JNLoader", "bin", "Release")
    copy_to_out(os.path.join(build_dir, "JNLoader.exe"))
    copy_to_out(os.path.join(build_dir, "JNLoader.exe.config"))

    require_in_out(["JNLoader.exe"])
    log("  JNLoader -> out/", "green")


def step_build_rlcomm():
    """Build RLCommPlugin -> out/RLCommPlugin.dll"""
    rlcomm_cs = os.path.join(RLCOMM_DIR, "RLCommPlugin.cs")
    if not os.path.isfile(rlcomm_cs):
        log("  WARNING: RLCommPlugin.cs not found", "yellow")
        return

    csc = find_csc()
    if not csc:
        log("  WARNING: csc.exe not found, skipping", "yellow")
        return

    out_dll = os.path.join(OUT_DIR, "RLCommPlugin.dll")
    refs = ["Cirnix.JassNative.Runtime.dll", "Cirnix.JassNative.dll",
            "Cirnix.JassNative.Common.dll", "EasyHook.dll", "Newtonsoft.Json.dll"]

    for ref in refs:
        if not os.path.isfile(os.path.join(OUT_DIR, ref)):
            log(f"  WARNING: {ref} not in out/, skipping RLCommPlugin", "yellow")
            return

    cmd = [csc, "/target:library", "/langversion:latest", f"/out:{out_dll}"]
    for ref in refs:
        cmd.append(f"/reference:{os.path.join(OUT_DIR, ref)}")
    cmd.append(rlcomm_cs)

    run_build(cmd, "RLCommPlugin.dll (csc)")
    require_in_out(["RLCommPlugin.dll"])
    log("  RLCommPlugin -> out/", "green")


# ============================================================
# War3Client Assembly (out/ -> War3Client/)
# ============================================================

def step_extract_war3():
    """Extract War3Client.zip"""
    if os.path.isdir(WAR3_DIR):
        log("  War3Client/ already exists", "green")
        return
    if not os.path.isfile(WAR3_ZIP):
        log("ERROR: War3Client.zip not found!", "red")
        sys.exit(1)
    log("  Extracting War3Client.zip...", "yellow")
    with zipfile.ZipFile(WAR3_ZIP, "r") as zf:
        zf.extractall(WAR3_DIR)
    log("  Extracted", "green")


def step_install_to_war3():
    """Copy out/ -> War3Client/ root (everything flat, no JNService)"""
    for f in os.listdir(OUT_DIR):
        src = os.path.join(OUT_DIR, f)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(WAR3_DIR, f))
    log("  out/ -> War3Client/ installed (flat)", "green")


def step_map():
    """Patch RL map"""
    map_dest = os.path.join(WAR3_DIR, "Maps", "rl")
    map_file = os.path.join(map_dest, "fateanother_rl.w3x")

    if os.path.isfile(map_file):
        log("  Map already exists", "green")
        return

    if platform.system() != "Windows":
        log("  WARNING: Map patching requires Windows. Skipping.", "yellow")
        return

    rl_patch = os.path.join(MAPPATCH_DIR, "rl_patch.py")
    if not os.path.isfile(rl_patch):
        log("  WARNING: rl_patch.py not found", "yellow")
        return

    log("  Patching RL map...", "yellow")
    result = subprocess.run([sys.executable, rl_patch], cwd=MAPPATCH_DIR,
                            capture_output=True, text=True)
    if result.returncode != 0:
        log("  Map patch failed!", "red")
        for line in (result.stdout + result.stderr).strip().split("\n")[-10:]:
            print(f"    {line}")
        sys.exit(1)

    patched = os.path.join(MAPPATCH_DIR, "fateanother_now.w3x")
    os.makedirs(map_dest, exist_ok=True)
    shutil.copy2(patched, map_file)
    log("  Map patched", "green")


def step_cleanup():
    for root, dirs, files in os.walk(WAR3_DIR):
        for f in files:
            if f == "CLAUDE.md" or f.endswith(".log"):
                os.remove(os.path.join(root, f))


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Build & assemble FateAnother RL")
    parser.add_argument("--skip-map", action="store_true", help="Skip map patching")
    parser.add_argument("--clean", action="store_true", help="Clean rebuild")
    parser.add_argument("--build-only", action="store_true", help="Build to out/ only")
    args = parser.parse_args()

    log("=" * 55, "cyan")
    log("  FateAnother RL - Build & Assemble", "cyan")
    log("=" * 55, "cyan")
    print()

    if args.clean:
        for d in [OUT_DIR, WAR3_DIR,
                  os.path.join(JASSNATIVE_DIR, "Build"),
                  os.path.join(JNLOADER_DIR, "JNLoader", "bin")]:
            if os.path.isdir(d):
                shutil.rmtree(d)
        log("[clean] All build artifacts removed", "green")
        print()

    msbuild = find_msbuild()
    if not msbuild:
        log("ERROR: MSBuild not found! Install Visual Studio 2019+", "red")
        sys.exit(1)
    log(f"  MSBuild: {msbuild}", "cyan")

    os.makedirs(OUT_DIR, exist_ok=True)

    # === BUILD PHASE (everything -> out/) ===
    print()
    log("[1/4] Third-party deps + EasyHook -> out/", "yellow")
    step_thirdparty()

    print()
    log("[2/4] Building JassNative...", "yellow")
    step_build_jassnative(msbuild)

    print()
    log("[3/4] Building JNLoader...", "yellow")
    step_build_jnloader(msbuild)

    print()
    log("[4/4] Building RLCommPlugin...", "yellow")
    step_build_rlcomm()

    # Show out/ contents
    print()
    log("  out/ contents:", "cyan")
    for f in sorted(os.listdir(OUT_DIR)):
        size = os.path.getsize(os.path.join(OUT_DIR, f))
        log(f"    {f:45s} {size:>10,} bytes")

    if args.build_only:
        print()
        log("=" * 55, "green")
        log("  Build complete! All outputs in out/", "green")
        log("=" * 55, "green")
        return

    # === ASSEMBLY PHASE (out/ -> War3Client/) ===
    print()
    log("[+] Assembling War3Client...", "yellow")
    step_extract_war3()
    step_install_to_war3()

    if not args.skip_map:
        step_map()

    step_cleanup()

    print()
    log("=" * 55, "green")
    log("  Build & assembly complete!", "green")
    log(f"  out/         <- all build outputs", "green")
    log(f"  War3Client/  <- ready to run", "green")
    log("=" * 55, "green")


if __name__ == "__main__":
    main()
