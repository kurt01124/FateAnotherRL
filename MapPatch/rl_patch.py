"""RL Patch - Bug fixes + Auto hero selection + RL training environment (JassNative TCP)"""
import ctypes, os, struct, shutil, re
from io import BytesIO

dll = ctypes.WinDLL(r".\StormLib_x64.dll")
dll.SFileOpenArchive.argtypes = [ctypes.c_wchar_p, ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
dll.SFileOpenArchive.restype = ctypes.c_bool
dll.SFileAddFileEx.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_char_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint]
dll.SFileAddFileEx.restype = ctypes.c_bool
dll.SFileOpenFileEx.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
dll.SFileOpenFileEx.restype = ctypes.c_bool
dll.SFileReadFile.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_uint), ctypes.c_void_p]
dll.SFileReadFile.restype = ctypes.c_bool
dll.SFileGetFileSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
dll.SFileGetFileSize.restype = ctypes.c_uint
dll.SFileCloseFile.argtypes = [ctypes.c_void_p]
dll.SFileCloseFile.restype = ctypes.c_bool
dll.SFileCloseArchive.argtypes = [ctypes.c_void_p]
dll.SFileCloseArchive.restype = ctypes.c_bool

MPQ_FILE_COMPRESS = 0x00000200
MPQ_FILE_REPLACEEXISTING = 0x80000000
MPQ_COMPRESSION_ZLIB = 0x02

base_path = r".\fateanother_original.w3x"
orig_path = r".\fateanother_verFS26A_original.w3x"

hOrig = ctypes.c_void_p()
dll.SFileOpenArchive(orig_path, 0, 0x100, ctypes.byref(hOrig))

def read_mpq(h, fn):
    hF = ctypes.c_void_p()
    if not dll.SFileOpenFileEx(h, fn, 0, ctypes.byref(hF)):
        return None
    sz = dll.SFileGetFileSize(hF, None)
    buf = ctypes.create_string_buffer(sz)
    rd = ctypes.c_uint(0)
    dll.SFileReadFile(hF, buf, sz, ctypes.byref(rd), None)
    d = buf.raw[:rd.value]
    dll.SFileCloseFile(hF)
    return d

orig_j = read_mpq(hOrig, b"war3map.j")
orig_misc = read_mpq(hOrig, b"war3mapMisc.txt")
orig_w3u = read_mpq(hOrig, b"war3map.w3u")
orig_w3a = read_mpq(hOrig, b"war3map.w3a")
orig_w3t = read_mpq(hOrig, b"war3map.w3t")
orig_w3i = read_mpq(hOrig, b"war3map.w3i")
dll.SFileCloseArchive(hOrig)

# Start with original JASS
j_data = orig_j

# ============================================================
# Patch 6: w3a A0A5 fix - add achd=0 (like A08F which works)
# ============================================================
def patch_w3a_fix_abilities(data, target_ids):
    """Add achd=0 to target abilities - A08F has this and works."""
    def make_int(fid, lvl, dp, val):
        return fid + struct.pack('<IIIiI', 0, lvl, dp, val, 0)

    result = data
    for target_id in target_ids:
        target_bytes = target_id.encode() if isinstance(target_id, str) else target_id

        f = BytesIO(result)
        f.read(4)  # version
        oc = struct.unpack('<I', f.read(4))[0]
        # Skip originals
        for _ in range(oc):
            f.read(8)
            mc = struct.unpack('<I', f.read(4))[0]
            for _ in range(mc):
                f.read(4)
                vt = struct.unpack('<I', f.read(4))[0]
                f.read(8)
                if vt == 3:
                    while f.read(1) != b'\x00': pass
                else:
                    f.read(4)
                f.read(4)
        cc = struct.unpack('<I', f.read(4))[0]

        found = False
        for _ in range(cc):
            f.read(4)  # base_id
            cid = f.read(4)
            mc_pos = f.tell()
            mc = struct.unpack('<I', f.read(4))[0]
            # Skip all mods
            for _ in range(mc):
                f.read(4)
                vt = struct.unpack('<I', f.read(4))[0]
                f.read(8)
                if vt == 3:
                    while f.read(1) != b'\x00': pass
                else:
                    f.read(4)
                f.read(4)
            mods_end = f.tell()

            if cid == target_bytes:
                # Add achd=0 (channel duration = 0, like A08F)
                achd_mod = make_int(b'achd', 0, 0, 0)

                tmp = bytearray(result)
                # Update mod_count
                new_mc = mc + 1
                struct.pack_into('<I', tmp, mc_pos, new_mc)
                # Insert achd mod at end of ability's modifications
                result = bytes(tmp[:mods_end]) + achd_mod + bytes(tmp[mods_end:])
                print(f"Patch 6: {target_id} added achd=0 (mods {mc}->{new_mc})")
                found = True
                break

        if not found:
            print(f"WARNING: {target_id} not found")

    return result

def patch_w3a_add_aord(data, aord_map):
    """Add aord (order string) field to abilities for RL agent order control.

    aord_map: dict of ability_id_str -> order_string
    Example: {"A01J": "thunderclap", "A02B": "divineshield"}
    """
    def make_str_mod(fid, val_str):
        """Create a string modification record for w3a."""
        val_bytes = val_str.encode('utf-8') + b'\x00'
        return fid + struct.pack('<III', 3, 0, 0) + val_bytes + struct.pack('<I', 0)

    result = data
    added = 0

    for target_id, order_str in aord_map.items():
        target_bytes = target_id.encode() if isinstance(target_id, str) else target_id

        f = BytesIO(result)
        f.read(4)  # version
        oc = struct.unpack('<I', f.read(4))[0]
        # Skip originals
        for _ in range(oc):
            f.read(8)
            mc = struct.unpack('<I', f.read(4))[0]
            for _ in range(mc):
                f.read(4)
                vt = struct.unpack('<I', f.read(4))[0]
                f.read(8)
                if vt == 3:
                    while f.read(1) != b'\x00': pass
                else:
                    f.read(4)
                f.read(4)
        cc = struct.unpack('<I', f.read(4))[0]

        found = False
        for _ in range(cc):
            f.read(4)  # base_id
            cid = f.read(4)
            mc_pos = f.tell()
            mc = struct.unpack('<I', f.read(4))[0]
            # Skip all mods
            for _ in range(mc):
                f.read(4)
                vt = struct.unpack('<I', f.read(4))[0]
                f.read(8)
                if vt == 3:
                    while f.read(1) != b'\x00': pass
                else:
                    f.read(4)
                f.read(4)
            mods_end = f.tell()

            if cid == target_bytes:
                aord_mod = make_str_mod(b'aord', order_str)
                tmp = bytearray(result)
                new_mc = mc + 1
                struct.pack_into('<I', tmp, mc_pos, new_mc)
                result = bytes(tmp[:mods_end]) + aord_mod + bytes(tmp[mods_end:])
                added += 1
                found = True
                break

        if not found:
            print(f"  WARNING: {target_id} not found in w3a for aord")

    return result, added

# A0A5 = Saber Instinct Enhancement, A052 = Lancer SwiftStrikes
w3a_data = patch_w3a_fix_abilities(orig_w3a, ['A0A5', 'A052'])

# ============================================================
# Patch RL-AORD: Add unique order strings to all RL abilities
# ============================================================
rl_aord_map = {
    # Saber H000
    "A087": "unavatar",          # Q InvisibleAir (→ANcl)
    "A01H": "unavengerform",     # W Caliburn (→ANcl)
    "A01J": "thunderclap",       # E Excalibur (AUcs, aord is backup)
    "A02B": "divineshield",      # R Avalon (→ANcl)
    "A0A5": "roar",              # D SaberInstinct (→ANcl)
    # Archer H001
    "A019": "transmute",         # Q Kanshou&Bakuya (→ANcl)
    "A01B": "windwalk",          # W BrokenPhantasm (→ANcl)
    "A014": "blight",            # E RhoAias (→ANcl)
    "A03C": "waterelemental",    # R UBW (→ANcl)
    # Lancer H002 (rune sub-abilities + main)
    "A035": "carrionscarabs",    # Q Rune-Ansuz (→ANcl)
    "A02T": "shockwave",         # D Rune-Ehwaz (→ANcl)
    "A03H": "mirrorimage",       # F Rune-Berkanan (→ANcl)
    "A05A": "stomp",             # rune Dagaz (sub-ability)
    "A05B": "earthquake",        # rune Eihwaz (sub-ability)
    "A052": "blizzard",          # W SwiftStrikes (→ANcl)
    "A01K": "web",               # E GaeBolg (→ANcl)
    "A028": "forceofnature",     # R FlyingSpear (→ANcl)
    # Rider H003
    "A01Q": "cyclone",           # Q CatenaSword (→ANcl)
    "A01R": "faeriefire",        # W BreakerGorgon (AUcs, aord is backup)
    "A01F": "metamorphosis",     # E BloodFort (→ANcl)
    "A01E": "inferno",           # R Bellerophon (→ANcl)
    # Caster/Medea H004
    "A049": "starfall",          # Q TerritoryCreation (→ANcl)
    "A08D": "aegis",             # W Aegis (ANcl, Ncl6 patched)
    # A08A already has aord="unbearform"
    "A06L": "carrionswarm",      # R HecaticGraea (→ANcl)
    # FakeAssassin H005
    "A01L": "deathcoil",         # Q Gatekeeper (→ANcl)
    "A05D": "darkportal",        # W Knowledge (→ANcl)
    "A01O": "fingerofdeath",     # E Windblade (→ANcl)
    "A01P": "firebolt",          # R TsubameGaeshi (→ANcl)
    # Berserker H006
    "A04J": "frostarmor",        # Q TrueStrike (AUcs, aord is backup)
    "A01D": "frostnova",         # W MadEnhancement (→ANcl)
    "A00Z": "sleep",             # E Bravery (→ANcl)
    "A015": "darkconversion",    # R NineLives (→ANcl)
    # SaberAlter H007
    "A00B": "cripple",           # Q Tyrant (→ANcl)
    "A07S": "recharge",          # W Vortigern (→ANcl)
    "A024": "possession",        # E PranaBurst (→ANcl)
    "A023": "animatedead",       # R ExcaliburMorgan (AUcs, aord is backup)
    # TrueAssassin H008
    "A09Y": "inferno",           # Q Steal (→ANcl)
    "A018": "deathanddecay",     # W SelfReconstruction (→ANcl)
    "A012": "ambush",            # E Ambush (AOwk, aord matches base)
    # A02A already has aord="unavengerform"
    # Gilgamesh H009
    "A01Y": "unavengerform",     # Q Marduk (→ANcl)
    "A01X": "creepthunderbolt",  # W Enkidu (→ANcl)
    "A01Z": "cloudoffog",        # E GateOfBabylon (→ANcl)
    "A02E": "controlmagic",      # R EnumaElish (AUcs, aord is backup)
    # Avenger H028 (CAST IDs, not learn IDs!)
    "A06J": "spellsteal",        # Q KillingIntent (→ANcl)
    "A05V": "polymorph",         # W TawrichZarich (→ANcl)
    "A05X": "drain",             # E Shade (→ANcl)
    "A02P": "absorb",            # R VergAvesta (→ANcl)
    # Lancelot H03M
    "A02Z": "charm",             # Q SubmachineGun (→ANcl)
    "A08F": "acidbomb",          # W DoubleEdgedSword (→ANcl)
    "A09B": "knightnotdie",      # E KnightNotDie (ANcl, Ncl6 patched)
    "A08S": "healingspray",      # R Arondight (→ANcl)
    # Diarmuid H04D (CAST IDs for E/R!)
    "A0AG": "transmute",         # Q Crash (→ANcl)
    "A0AI": "lavamonster",       # W DoubleSpearMastery (→ANcl)
    "A0AL": "soulburn",          # E GaeBuidhe (→ANcl)
    "A0AM": "volcano",           # R GaeDearg (→ANcl)
    # Seal
    "A094": "sealact",           # CommandSeal (ANcl, Ncl6 patched)
}
w3a_data, aord_count = patch_w3a_add_aord(w3a_data, rl_aord_map)
print(f"Patch RL-AORD: Added aord to {aord_count}/{len(rl_aord_map)} abilities")

# ============================================================
# Patch RL-NCL6: Modify Channel base order strings for collision resolution
# ============================================================
def patch_w3a_modify_ncl6(data, ncl6_map):
    """Modify existing Ncl6 (Channel base order) strings in w3a.

    For ANcl (Channel) abilities, the actual order string is Ncl6, NOT aord.
    Abilities sharing the same Ncl6 on the same hero cause order collisions.
    This patches Ncl6 to give each ability a unique order string.

    ncl6_map: {ability_id_str: new_ncl6_value}
    """
    output = BytesIO()
    pos = 0
    modified = 0

    # version
    output.write(data[pos:pos+4]); pos += 4

    for table_idx in range(2):
        count = struct.unpack_from('<I', data, pos)[0]
        output.write(data[pos:pos+4]); pos += 4

        for _ in range(count):
            oid = data[pos:pos+4]; output.write(oid); pos += 4
            cid = data[pos:pos+4]; output.write(cid); pos += 4
            mc = struct.unpack_from('<I', data, pos)[0]
            output.write(data[pos:pos+4]); pos += 4

            key = cid.decode('latin-1') if cid != b'\x00\x00\x00\x00' else oid.decode('latin-1')

            for _ in range(mc):
                mod_start = pos
                fid = data[pos:pos+4]; pos += 4
                dt = struct.unpack_from('<I', data, pos)[0]; pos += 4
                level = struct.unpack_from('<I', data, pos)[0]; pos += 4
                variation = struct.unpack_from('<I', data, pos)[0]; pos += 4

                if dt == 3:
                    val_start = pos
                    end = data.index(b'\x00', pos)
                    old_str = data[val_start:end].decode('utf-8', errors='replace')
                    pos = end + 1
                else:
                    pos += 4
                pos += 4  # end marker
                mod_end = pos

                if key in ncl6_map and fid == b'Ncl6' and dt == 3:
                    new_val = ncl6_map[key]
                    output.write(fid)
                    output.write(struct.pack('<III', 3, level, variation))
                    output.write(new_val.encode('utf-8') + b'\x00')
                    output.write(struct.pack('<I', 0))
                    modified += 1
                    if modified <= 6:
                        print(f"  Ncl6: {key} lvl={level} '{old_str}' -> '{new_val}'")
                else:
                    output.write(data[mod_start:mod_end])

    return output.getvalue(), modified

# Resolve Channel ability order string collisions
# A08D, A09B, A094 all share Ncl6="channel" — give each a unique order
rl_ncl6_map = {
    "A08D": "aegis",        # Caster W - Aegis (was "channel")
    "A09B": "knightnotdie", # Lancelot E - KnightNotDie (was "channel")
    "A094": "sealact",      # Seal Activate for ALL heroes (was "channel")
}
w3a_data, ncl6_count = patch_w3a_modify_ncl6(w3a_data, rl_ncl6_map)
print(f"Patch RL-NCL6: Modified Ncl6 for {ncl6_count} entries (3 abilities)")

# ============================================================
# Patch RL-ANCL: Convert non-working ability base types to ANcl
# ============================================================
# IssueOrder JASS natives only work for ANcl (via Ncl6), AOwk (via aord),
# and AUcs (via "channel"). All other base types ignore order strings.
# Solution: Change base type (orig_id) to ANcl and set Ncl6 order string.
def patch_w3a_convert_to_ancl(data, convert_map):
    """Convert ability base types to ANcl (Channel) for IssueOrder compatibility.

    For each ability in convert_map:
    1. Changes orig_id to 'ANcl'
    2. Adds Ncl6 = order_string (for IssueOrder matching)
    3. Adds Ncl1 = 0.0 (art duration)
    4. Adds Ncl3 = 1 (options: visible)
    5. Adds Ncl5 = 0 (base order ID)

    convert_map: dict of ability_id_str -> ncl6_order_string
    """
    NUM_LEVELS = 5  # WC3 Channel fields must be per-level (1-5)

    def make_str_mod_lvl(fid, lvl, val_str):
        val_bytes = val_str.encode('utf-8') + b'\x00'
        return fid + struct.pack('<III', 3, lvl, 0) + val_bytes + struct.pack('<I', 0)

    def make_real_mod_lvl(fid, lvl, val):
        return fid + struct.pack('<III', 1, lvl, 0) + struct.pack('<f', val) + struct.pack('<I', 0)

    def make_int_mod_lvl(fid, lvl, val):
        return fid + struct.pack('<IIIiI', 0, lvl, 0, val, 0)

    output = BytesIO()
    pos = 0
    converted = 0

    # version
    output.write(data[pos:pos+4]); pos += 4

    for table_idx in range(2):  # original abilities + custom abilities
        count = struct.unpack_from('<I', data, pos)[0]
        output.write(data[pos:pos+4]); pos += 4

        for _ in range(count):
            oid = data[pos:pos+4]; pos += 4
            cid = data[pos:pos+4]; pos += 4
            mc = struct.unpack_from('<I', data, pos)[0]; pos += 4

            # Read all modification bytes for this ability
            mods_start = pos
            for _ in range(mc):
                pos += 4  # field_id
                vt = struct.unpack_from('<I', data, pos)[0]; pos += 4
                pos += 8  # level + data_point
                if vt == 3:
                    while data[pos:pos+1] != b'\x00': pos += 1
                    pos += 1
                else:
                    pos += 4
                pos += 4  # end marker
            mods_end = pos
            mods_bytes = data[mods_start:mods_end]

            key = cid.decode('latin-1')
            if key in convert_map:
                ncl6_val = convert_map[key]
                old_base = oid.decode('latin-1')
                # Write with changed orig_id
                output.write(b'ANcl')           # Changed base type
                output.write(cid)
                # 5 fields × 5 levels = 25 new modifications
                added_mods = NUM_LEVELS * 5     # Ncl1, Ncl3, Ncl4, Ncl5, Ncl6 × 5 levels
                new_mc = mc + added_mods
                output.write(struct.pack('<I', new_mc))
                output.write(mods_bytes)         # Existing modifications
                # Add Channel fields for EACH level (1-5)
                for lvl in range(1, NUM_LEVELS + 1):
                    output.write(make_str_mod_lvl(b'Ncl6', lvl, ncl6_val))
                    output.write(make_real_mod_lvl(b'Ncl1', lvl, 0.0))
                    output.write(make_int_mod_lvl(b'Ncl3', lvl, 1))
                    output.write(make_real_mod_lvl(b'Ncl4', lvl, 0.0))
                    output.write(make_int_mod_lvl(b'Ncl5', lvl, 0))
                converted += 1
                print(f"  {key}: {old_base} -> ANcl, Ncl6='{ncl6_val}' (mods {mc}->{new_mc})")
            else:
                # Write unchanged
                output.write(oid)
                output.write(cid)
                output.write(struct.pack('<I', mc))
                output.write(mods_bytes)

    print(f"Patch RL-ANCL: Converted {converted}/{len(convert_map)} abilities to ANcl")
    return output.getvalue()

# All abilities with non-working base types (not ANcl/AOwk/AUcs)
# Key = ability ID, Value = Ncl6 order string (matches aord/HeroDataTable)
rl_ancl_convert_map = {
    # Saber H000
    "A087": "unavatar",          # Q InvisibleAir (was ANcl pre-existing)
    "A01H": "unavengerform",     # W Caliburn (was ANcl pre-existing)
    "A02B": "divineshield",      # R Avalon (was ANbr)
    "A0A5": "roar",              # D SaberInstinct (was Absk)
    # Archer H001
    "A019": "transmute",         # Q Kanshou&Bakuya (was ANcl pre-existing)
    "A01B": "windwalk",          # W BrokenPhantasm (was AOwk)
    "A014": "blight",            # E RhoAias (was AHtb)
    "A03C": "waterelemental",    # R UBW (was AEfk)
    # Lancer H002
    "A035": "carrionscarabs",    # Q Rune-Ansuz (was AUcb)
    "A052": "blizzard",          # W SwiftStrikes (was Absk)
    "A01K": "web",               # E GaeBolg (was Aweb)
    "A028": "forceofnature",     # R FlyingSpear (was ANcs)
    "A02T": "shockwave",         # D Rune-Ehwaz (was AIh2)
    "A03H": "mirrorimage",       # F Rune-Berkanan (was Asta)
    # Rider H003
    "A01Q": "cyclone",           # Q CatenaSword (was AEfk)
    "A01F": "metamorphosis",     # E BloodFort (was ANbr)
    "A01E": "inferno",           # R Bellerophon (was AUin)
    # Caster H004
    "A049": "starfall",          # Q TerritoryCreation (was AIbt)
    "A06L": "carrionswarm",      # R HecaticGraea (was ANhs)
    # FakeAssassin H005
    "A01L": "deathcoil",         # Q Gatekeeper (was ANcr)
    "A05D": "darkportal",        # W Knowledge (was ANbr)
    "A01O": "fingerofdeath",     # E Windblade (was AHtc)
    "A01P": "firebolt",          # R TsubameGaeshi (was ANfd)
    # Berserker H006
    "A01D": "frostnova",         # W MadEnhancement (was ANbr)
    "A00Z": "sleep",             # E Bravery (was Absk)
    "A015": "darkconversion",    # R NineLives (was ANcs)
    # SaberAlter H007
    "A00B": "cripple",           # Q Tyrant (was Aroa)
    "A07S": "recharge",          # W Vortigern (was Ambt)
    "A024": "possession",        # E PranaBurst (was AHtc)
    # TrueAssassin H008
    "A09Y": "inferno",           # Q Steal (was ANfd)
    "A018": "deathanddecay",     # W SelfReconstruction (was AEsb)
    # Gilgamesh H009
    "A01Y": "unavengerform",     # Q Marduk (was ANcl pre-existing)
    "A01X": "creepthunderbolt",  # W Enkidu (was Amls)
    "A01Z": "cloudoffog",        # E GateOfBabylon (was ANst)
    # Avenger H028
    "A06J": "spellsteal",        # Q KillingIntent (was ACro)
    "A05V": "polymorph",         # W TawrichZarich (was ACpy)
    "A05X": "drain",             # E Shade (was ANcr)
    "A02P": "absorb",            # R VergAvesta (was Aabs)
    # Lancelot H03M
    "A02Z": "charm",             # Q SubmachineGun (was ANcs)
    "A08F": "acidbomb",          # W DoubleEdgedSword (was Absk)
    "A08S": "healingspray",      # R Arondight (was ANcr)
    # Diarmuid H04D
    "A0AG": "transmute",         # Q Crash (was AOcl)
    "A0AI": "lavamonster",       # W DoubleSpearMastery (was AOw2)
    "A0AL": "soulburn",          # E GaeBuidhe (was AUfn)
    "A0AM": "volcano",           # R GaeDearg (was Afod)
}
w3a_data = patch_w3a_convert_to_ancl(w3a_data, rl_ancl_convert_map)

# ============================================================
# Patch 15: AntiMapHack removal (DeSync fix)
# ============================================================
def remove_antimaphack(jass_bytes):
    """AntiMapHack 시스템 제거 - DeSync 버그 수정"""
    content = jass_bytes.decode('utf-8', errors='replace')
    lines = content.split('\n')
    modified = []
    skip_until_end = False
    removed_count = 0

    for line in lines:
        # 함수 시작 - 제거 대상
        if not skip_until_end:
            match = re.match(r'^function (CameraX|CameraY|AntiMapHack___\w+) takes', line)
            if match:
                func_name = match.group(1)
                removed_count += 1
                # 빈 함수로 교체
                if func_name == 'CameraX':
                    modified.append('function CameraX takes player p returns real')
                    modified.append('return 0')
                    modified.append('endfunction')
                elif func_name == 'CameraY':
                    modified.append('function CameraY takes player p returns real')
                    modified.append('return 0')
                    modified.append('endfunction')
                else:
                    modified.append(line)
                    modified.append('endfunction')
                skip_until_end = True
                continue

        if skip_until_end:
            if line.strip() == 'endfunction':
                skip_until_end = False
            continue

        # AntiMapHack___OnInit 호출 제거
        if 'AntiMapHack___OnInit' in line and 'ExecuteFunc' in line:
            modified.append('// [PATCH15] AntiMapHack removed - DeSync fix')
            continue

        modified.append(line)

    return '\n'.join(modified).encode('utf-8'), removed_count

j_data, amh_count = remove_antimaphack(j_data)
print(f"Patch 15: AntiMapHack removed ({amh_count} functions) - DeSync fix OK")

# ============================================================
# Patch 26: AntiMapHack 잔재 완전 정리
# ============================================================
# 26a: IsMapHack → always return false
old_ismaphack = "function IsMapHack takes nothing returns boolean\nreturn AntiMapHack___mH\nendfunction".encode('utf-8')
new_ismaphack = "function IsMapHack takes nothing returns boolean\nreturn false\nendfunction".encode('utf-8')
assert old_ismaphack in j_data, "IsMapHack function not found"
j_data = j_data.replace(old_ismaphack, new_ismaphack, 1)
print("Patch 26a: IsMapHack -> always false OK")

# 26b: PlayerLeave maphack check removed
old_leave = ('if(AntiMapHack___mH)then\n'
             'call Text(s__User_originalName[user]+"\xea\xbb\x98\xec\x84\x9c \xed\x87\xb4\xec\x9e\xa5\xed\x96\x88\xec\x8a\xb5\xeb\x8b\x88\xeb\x8b\xa4 (\xeb\xa7\xb5\xed\x95\xb5 \xec\x82\xac\xec\x9a\xa9)",5)\n'
             'else\n'
             'call Text(s__User_originalName[user]+"\xea\xbb\x98\xec\x84\x9c \xeb\x82\x98\xea\xb0\x80\xec\x85\xa8\xec\x8a\xb5\xeb\x8b\x88\xeb\x8b\xa4.",5)\n'
             'endif').encode('raw_unicode_escape')
new_leave = 'call Text(s__User_originalName[user]+"\xea\xbb\x98\xec\x84\x9c \xeb\x82\x98\xea\xb0\x80\xec\x85\xa8\xec\x8a\xb5\xeb\x8b\x88\xeb\x8b\xa4.",5)'.encode('raw_unicode_escape')
if old_leave in j_data:
    j_data = j_data.replace(old_leave, new_leave, 1)
    print("Patch 26b: PlayerLeave maphack check removed OK")
else:
    print("Patch 26b: PlayerLeave maphack check - trying UTF-8 pattern")
    old_leave2 = ('if(AntiMapHack___mH)then\n'
                  'call Text(s__User_originalName[user]+"께서 퇴장했습니다 (맵핵 사용)",5)\n'
                  'else\n'
                  'call Text(s__User_originalName[user]+"께서 나가셨습니다.",5)\n'
                  'endif').encode('utf-8')
    new_leave2 = 'call Text(s__User_originalName[user]+"께서 나가셨습니다.",5)'.encode('utf-8')
    assert old_leave2 in j_data, "PlayerLeave AntiMapHack check not found in either encoding"
    j_data = j_data.replace(old_leave2, new_leave2, 1)
    print("Patch 26b: PlayerLeave maphack check removed (UTF-8) OK")

# ============================================================
# Patch 27: Knockback div-by-zero guards
# ============================================================
# 27a: Knockback_start
old_kb_start = b"set s__Knockback_c[this]=R2I(32*dur)\nset s__Knockback_cos[this]=dist/s__Knockback_c[this]*Cos(a)\nset s__Knockback_sin[this]=dist/s__Knockback_c[this]*Sin(a)"
new_kb_start = b"set s__Knockback_c[this]=R2I(32*dur)\nif s__Knockback_c[this]<1 then\nset s__Knockback_c[this]=1\nendif\nset s__Knockback_cos[this]=dist/s__Knockback_c[this]*Cos(a)\nset s__Knockback_sin[this]=dist/s__Knockback_c[this]*Sin(a)"
assert old_kb_start in j_data, "Knockback_start div pattern not found"
j_data = j_data.replace(old_kb_start, new_kb_start, 1)
print("Patch 27a: Knockback_start div-by-zero guard OK")

# 27b: Knockback_startEx
old_kb_ex = b"set s__Knockback_c[this]=R2I(32*dur)\nset s__Knockback_cos[this]=d/s__Knockback_c[this]*Cos(a)\nset s__Knockback_sin[this]=d/s__Knockback_c[this]*Sin(a)"
new_kb_ex = b"set s__Knockback_c[this]=R2I(32*dur)\nif s__Knockback_c[this]<1 then\nset s__Knockback_c[this]=1\nendif\nset s__Knockback_cos[this]=d/s__Knockback_c[this]*Cos(a)\nset s__Knockback_sin[this]=d/s__Knockback_c[this]*Sin(a)"
assert old_kb_ex in j_data, "Knockback_startEx div pattern not found"
j_data = j_data.replace(old_kb_ex, new_kb_ex, 1)
print("Patch 27b: Knockback_startEx div-by-zero guard OK")

# ============================================================
# Patch 28: ZeroAssassin BattleRetreat fix
# ============================================================
old_za_bug = b"s__Cooldown_check((GetPlayerId(GetOwningPlayer((s__Chulainn_unit)))),'I02A')"
new_za_fix = b"s__Cooldown_check((GetPlayerId(GetOwningPlayer((s__ZeroAssassin_unit)))),'I02A')"
assert old_za_bug in j_data, "ZeroAssassin Battle Retreat copy-paste bug not found"
j_data = j_data.replace(old_za_bug, new_za_fix, 1)
print("Patch 28: ZeroAssassin BattleRetreat Chulainn->ZeroAssassin fix OK")

# ============================================================
# Patch 29: Avenger doubleAttack alive check
# ============================================================
old_avenger = b"function s__Avenger_onExpiration takes nothing returns nothing\ncall ReleaseTimer(GetExpiredTimer())\ncall UnitDamageTargetEx(s__Avenger_unit,(UnitIndexer___e[(s__Avenger_doubleAttackTarget)])"
new_avenger = b"function s__Avenger_onExpiration takes nothing returns nothing\ncall ReleaseTimer(GetExpiredTimer())\nif s__Avenger_doubleAttackTarget==0 or not UnitAlive((UnitIndexer___e[(s__Avenger_doubleAttackTarget)]))then\nset s__Avenger_doubleAttackTarget=0\nreturn\nendif\ncall UnitDamageTargetEx(s__Avenger_unit,(UnitIndexer___e[(s__Avenger_doubleAttackTarget)])"
assert old_avenger in j_data, "Avenger onExpiration pattern not found"
j_data = j_data.replace(old_avenger, new_avenger, 1)
print("Patch 29: Avenger doubleAttack alive check OK")

# ============================================================
# Patch 30: Karna periodicBurn alive check
# ============================================================
old_karna_burn2 = b"function s__KavachaAndKundala_Skill_periodicBurn takes nothing returns nothing\nlocal real x=GetUnitX(s__Karna_unit)\nlocal real y=GetUnitY(s__Karna_unit)\nlocal real radius=KavachaAndKundala__BURN_AOE\nlocal unit u"
new_karna_burn2 = b"function s__KavachaAndKundala_Skill_periodicBurn takes nothing returns nothing\nlocal real x\nlocal real y\nlocal real radius\nlocal unit u\nif not UnitAlive(s__Karna_unit)then\nreturn\nendif\nset x=GetUnitX(s__Karna_unit)\nset y=GetUnitY(s__Karna_unit)\nset radius=KavachaAndKundala__BURN_AOE"
if old_karna_burn2 in j_data:
    j_data = j_data.replace(old_karna_burn2, new_karna_burn2, 1)
    print("Patch 30: Karna periodicBurn alive check OK")
else:
    print("Patch 30: Karna periodicBurn pattern not found - SKIPPED")

# ============================================================
# Patch 31: BloodBath alive check (FIXED - all 10 locals)
# ============================================================
old_bloodbath = b"function s__BloodBath_periodic takes nothing returns nothing\nlocal integer i=0\nlocal integer j\nlocal real x=GetUnitX(s__BloodBath_caster)\nlocal real y=GetUnitY(s__BloodBath_caster)\nlocal real dx=0\nlocal real dy=0\nlocal real calc_y=0\nlocal real angle=bj_PI/4\nlocal real d\nlocal unit u\nset s__BloodBath_r=s__BloodBath_r+(BloodBath__AOE/BloodBath__DIVISION_POINTS)"
new_bloodbath = b"function s__BloodBath_periodic takes nothing returns nothing\nlocal integer i\nlocal integer j\nlocal real x\nlocal real y\nlocal real dx\nlocal real dy\nlocal real calc_y\nlocal real angle\nlocal real d\nlocal unit u\nif not UnitAlive(s__BloodBath_caster)then\nreturn\nendif\nset i=0\nset x=GetUnitX(s__BloodBath_caster)\nset y=GetUnitY(s__BloodBath_caster)\nset dx=0\nset dy=0\nset calc_y=0\nset angle=bj_PI/4\nset s__BloodBath_r=s__BloodBath_r+(BloodBath__AOE/BloodBath__DIVISION_POINTS)"
if old_bloodbath in j_data:
    j_data = j_data.replace(old_bloodbath, new_bloodbath, 1)
    print("Patch 31: BloodBath alive check OK")
else:
    print("Patch 31: BloodBath pattern not found - SKIPPED")

# ============================================================
# Patch 32: Berserker GodHand null guard
# ============================================================
old_berserker_exp = b"function s__Berserker_onExpiration takes nothing returns nothing\nlocal real x=GetUnitX(s__Berserker_unit)\nlocal real y=GetUnitY(s__Berserker_unit)\nset s__Berserker_godHandCount=s__Berserker_godHandCount-1\nset s__Berserker_c=0\ncall ReleaseTimer(GetExpiredTimer())"
new_berserker_exp = b"function s__Berserker_onExpiration takes nothing returns nothing\nlocal real x\nlocal real y\ncall ReleaseTimer(GetExpiredTimer())\nif GetUnitTypeId(s__Berserker_unit)==0 then\nreturn\nendif\nset x=GetUnitX(s__Berserker_unit)\nset y=GetUnitY(s__Berserker_unit)\nset s__Berserker_godHandCount=s__Berserker_godHandCount-1\nset s__Berserker_c=0"
if old_berserker_exp in j_data:
    j_data = j_data.replace(old_berserker_exp, new_berserker_exp, 1)
    print("Patch 32: Berserker GodHand onExpiration null guard OK")
else:
    print("Patch 32: Berserker GodHand pattern not found - SKIPPED")

# ============================================================
# Patch 16: FFA scoreboard fix
# ============================================================
old_board = b"set s__User_ownBoardPos[((i))]=(1+i)"
new_board = b"set s__User_ownBoardPos[((i))]=(1+s__Game_playerCount)"
assert old_board in j_data, "FFA board position pattern not found"
j_data = j_data.replace(old_board, new_board)
ffa_count = j_data.count(new_board)
print(f"Patch 16: FFA scoreboard fix - sequential position ({ffa_count} replacements)")

# ============================================================
# Patch RL-0: Allow computer players to be "Playing"
# ============================================================
old_playing_check = b"set Playing[i]=GetPlayerSlotState(P[i])==PLAYER_SLOT_STATE_PLAYING and GetPlayerController(P[i])==MAP_CONTROL_USER"
new_playing_check = b"set Playing[i]=GetPlayerSlotState(P[i])==PLAYER_SLOT_STATE_PLAYING"
assert old_playing_check in j_data, "Playing[] controller check not found"
j_data = j_data.replace(old_playing_check, new_playing_check, 1)
print("Patch RL-0: Computer players included in Playing[] OK")

# ============================================================
# NEW RL PATCH: Auto hero selection - sequential assignment
# ============================================================
# Replace allRandom to use fixed assignment
old_all_random = b"function s__HeroSelection_allRandom takes nothing returns nothing\nlocal integer i=0\nloop\nif Playing[i]and s__User_hero[(i)]==0 then\ncall s__Hero_register((GetUnitUserData((CreateUnit(P[(i)],(s__HeroPool_id[GetRandomInt(1,s__HeroPool_max)]),GetRectCenterX(gg_rct_MS3),GetRectCenterY(gg_rct_MS3),270)))),true)\nendif\nset i=i+1\nexitwhen i==12\nendloop\nendfunction"
new_all_random = (
    b"function s__HeroSelection_allRandom takes nothing returns nothing\n"
    b"local integer i=0\n"
    b"loop\n"
    b"if Playing[i]and s__User_hero[(i)]==0 then\n"
    b"call s__Hero_register((GetUnitUserData((CreateUnit(P[(i)],(s__HeroPool_id[i+1]),0.0,0.0,270)))),true)\n"
    b"endif\n"
    b"set i=i+1\n"
    b"exitwhen i==12\n"
    b"endloop\n"
    b"endfunction"
)
assert old_all_random in j_data, "allRandom function not found"
j_data = j_data.replace(old_all_random, new_all_random, 1)
print("Patch RL-1: allRandom sequential + spread OK")

# Patch RL-1b: sa__ trigger condition version (THIS is what actually runs via TriggerEvaluate)
old_sa_all_random = b"function sa__HeroSelection_allRandom takes nothing returns boolean\nlocal integer i=0\nloop\nif Playing[i]and s__User_hero[(i)]==0 then\ncall s__Hero_register((GetUnitUserData((CreateUnit(P[(i)],(s__HeroPool_id[GetRandomInt(1,s__HeroPool_max)]),GetRectCenterX(gg_rct_MS3),GetRectCenterY(gg_rct_MS3),270)))),true)\nendif\nset i=i+1\nexitwhen i==12\nendloop\nreturn true\nendfunction"
new_sa_all_random = (
    b"function sa__HeroSelection_allRandom takes nothing returns boolean\n"
    b"local integer i=0\n"
    b"loop\n"
    b"if Playing[i]and s__User_hero[(i)]==0 then\n"
    b"call s__Hero_register((GetUnitUserData((CreateUnit(P[(i)],(s__HeroPool_id[i+1]),0.0,0.0,270)))),true)\n"
    b"endif\n"
    b"set i=i+1\n"
    b"exitwhen i==12\n"
    b"endloop\n"
    b"return true\n"
    b"endfunction"
)
assert old_sa_all_random in j_data, "sa__HeroSelection_allRandom trigger condition not found"
j_data = j_data.replace(old_sa_all_random, new_sa_all_random, 1)
print("Patch RL-1b: sa__allRandom trigger condition sequential + spread OK")

# Replace createRandom to use fixed assignment
old_create_random = b"call s__Hero_register((GetUnitUserData((CreateUnit(P[playerId],(s__HeroPool_id[GetRandomInt(1,s__HeroPool_max)]),GetRectCenterX(gg_rct_MS3),GetRectCenterY(gg_rct_MS3),270)))),true)"
new_create_random = b"call s__Hero_register((GetUnitUserData((CreateUnit(P[playerId],(s__HeroPool_id[playerId+1]),GetRectCenterX(gg_rct_MS3),GetRectCenterY(gg_rct_MS3),270)))),true)"
if old_create_random in j_data:
    j_data = j_data.replace(old_create_random, new_create_random, 1)
    print("Patch RL-2: createRandom sequential assignment OK")
else:
    print("WARNING: createRandom pattern not found - SKIPPED")

# Replace onSelect random with fixed assignment
old_onselect = b"call s__Hero_register((GetUnitUserData((CreateUnit(P[(playerId)],(s__HeroPool_id[GetRandomInt(1,s__HeroPool_max)]),GetRectCenterX(gg_rct_MS3),GetRectCenterY(gg_rct_MS3),270)))),true)"
new_onselect = b"call s__Hero_register((GetUnitUserData((CreateUnit(P[(playerId)],(s__HeroPool_id[playerId+1]),GetRectCenterX(gg_rct_MS3),GetRectCenterY(gg_rct_MS3),270)))),true)"
# Count how many times this appears (should be at least 1 for onSelect function)
onselect_count = j_data.count(old_onselect)
if onselect_count > 0:
    j_data = j_data.replace(old_onselect, new_onselect)
    print(f"Patch RL-3: onSelect sequential assignment OK ({onselect_count} replacements)")
else:
    print("WARNING: onSelect pattern not found - SKIPPED")

# ============================================================
# Patch RL-3a: Expand RespawnZone from 10 → 12 (using StartLocation coords)
# Original game was 5v5 (10 zones), now 6v6 needs 12
# ============================================================
old_zones = (
    b"call s__RespawnZone_addZone(7840,2240,8096,2464)\n"
    b"call s__RespawnZone_addZone(-1536,3808,-1280,4064)\n"
    b"call s__RespawnZone_addZone(0,-224,256,32)\n"
    b"call s__RespawnZone_addZone(-6400,-1440,-6112,-1216)\n"
    b"call s__RespawnZone_addZone(512,4640,768,4864)\n"
    b"call s__RespawnZone_addZone(-7168,1984,-7040,2240)\n"
    b"call s__RespawnZone_addZone(-7104,4960,-6680,5216)\n"
    b"call s__RespawnZone_addZone(3328,-1408,3552,-1152)\n"
    b"call s__RespawnZone_addZone(6656,3904,6912,4160)\n"
    b"call s__RespawnZone_addZone(7936,640,8192,896)"
)
# Original 10 zones + 2 user-verified via World Editor = 12 total
new_zones = (
    b"call s__RespawnZone_addZone(7840,2240,8096,2464)\n"      # zone 1: 우 집 위
    b"call s__RespawnZone_addZone(-1536,3808,-1280,4064)\n"     # zone 2: 좌 공원
    b"call s__RespawnZone_addZone(0,-224,256,32)\n"             # zone 3: 좌 다리 아래
    b"call s__RespawnZone_addZone(-6400,-1440,-6112,-1216)\n"   # zone 4: 아인츠베른
    b"call s__RespawnZone_addZone(512,4640,768,4864)\n"         # zone 5: 좌 축구골대
    b"call s__RespawnZone_addZone(-7168,1984,-7040,2240)\n"     # zone 6: 류도사
    b"call s__RespawnZone_addZone(-7104,4960,-6680,5216)\n"     # zone 7: 에미야 뒤
    b"call s__RespawnZone_addZone(3328,-1408,3552,-1152)\n"     # zone 8: 우 묘지 옆
    b"call s__RespawnZone_addZone(6656,3904,6912,4160)\n"       # zone 9: 우 숲
    b"call s__RespawnZone_addZone(7936,640,8192,896)\n"         # zone 10: 우 집 아래
    b"call s__RespawnZone_addZone(2316,3711,2572,3967)\n"       # zone 11: 우 항구 (NEW)
    b"call s__RespawnZone_addZone(7944,1709,8200,1965)"         # zone 12: 우 집 중간 (NEW)
)
assert old_zones in j_data, "RespawnZone addZone calls not found"
j_data = j_data.replace(old_zones, new_zones, 1)
print("Patch RL-3a: RespawnZone 10 -> 12 (user-verified) OK")

# ============================================================
# Patch RL-3b: HeroPool order → fixed team composition
# Team1(P0-5): Saber,Archer,Lancer,TrueAssassin,Avenger,Diarmuid
# Team2(P6-11): Berserker,Rider,FakeAssassin,SaberAlter,Gilgamesh,Lancelot
# ============================================================
old_pool = (
    b"set s__HeroPool_max=0\n"
    b"call s__HeroPool_add('H000')\n"
    b"call s__HeroPool_add('H001')\n"
    b"call s__HeroPool_add('H002')\n"
    b"call s__HeroPool_add('H003')\n"
    b"call s__HeroPool_add('H004')\n"
    b"call s__HeroPool_add('H005')\n"
    b"call s__HeroPool_add('H006')\n"
    b"call s__HeroPool_add('H007')\n"
    b"call s__HeroPool_add('H008')\n"
    b"call s__HeroPool_add('H009')\n"
    b"call s__HeroPool_add('H028')\n"
    b"call s__HeroPool_add('H03M')\n"
    b"call s__HeroPool_add('H04D')\n"
    b"call s__HeroPool_add('H00I')\n"
    b"call s__HeroPool_add('E002')\n"
    b"call s__HeroPool_add('H00A')"
)
new_pool = (
    b"set s__HeroPool_max=0\n"
    b"call s__HeroPool_add('H000')\n"  # P0  Saber
    b"call s__HeroPool_add('H001')\n"  # P1  Archer
    b"call s__HeroPool_add('H002')\n"  # P2  Lancer
    b"call s__HeroPool_add('H008')\n"  # P3  TrueAssassin
    b"call s__HeroPool_add('H028')\n"  # P4  Avenger
    b"call s__HeroPool_add('H03M')\n"  # P5  Lancelot
    b"call s__HeroPool_add('H006')\n"  # P6  Berserker
    b"call s__HeroPool_add('H003')\n"  # P7  Rider
    b"call s__HeroPool_add('H005')\n"  # P8  FakeAssassin
    b"call s__HeroPool_add('H007')\n"  # P9  SaberAlter
    b"call s__HeroPool_add('H009')\n"  # P10 Gilgamesh
    b"call s__HeroPool_add('H04D')"    # P11 Diarmuid
)
assert old_pool in j_data, "HeroPool init not found"
j_data = j_data.replace(old_pool, new_pool, 1)
print("Patch RL-3b: HeroPool fixed order (12 heroes, planned teams) OK")

# ============================================================
# Patch RL-4: DISABLED - native declarations break JASS compiler
# JassNative custom natives cannot be declared in JASS source.
# UnitAlive works because it's a hidden WC3 engine native.
# JNRLSendState etc are runtime-only (added by C# plugin hook).
# Communication will be done entirely from C# side instead.
# ============================================================
# Patch RL-4: native RLRand declaration (provided by RLCommPlugin.dll)
old_globals = b"globals"
# Find first occurrence of "globals" keyword (at top of file)
native_decl = b"native RLRand takes integer min, integer max returns integer\nnative JNSetSyncDelay takes integer delay returns nothing\nnative JNGetSyncDelay takes nothing returns integer\nnative JNGetConnectionState takes nothing returns integer\nnative JNWriteLog takes string msg returns nothing\nnative RLSetPDAT takes integer pid, integer u0, integer u1, integer u2, integer u3, integer u4, integer u5, integer u6, integer u7, integer u8, integer sealCd, integer sealActive, integer sealFirstCd, integer attrCount returns nothing\nnative RLSetScore takes integer t1, integer t2 returns nothing\nnative RLTick takes integer tick returns nothing\n"
idx_globals = j_data.index(old_globals)
j_data = j_data[:idx_globals] + native_decl + j_data[idx_globals:]
print("Patch RL-4: native RLRand declaration OK")

# ============================================================
# Patch RL-5: RL global variables (before endglobals)
# ============================================================
old_endg = b"""player f__result_player
endglobals"""
new_endg = b"""player f__result_player
integer rl_h1_hid
real array rl_spawn_lx
real array rl_spawn_ly
real array rl_spawn_rx
real array rl_spawn_ry
integer rl_team1_side
integer array rl_perm
integer rl_tick_mod
endglobals"""
assert old_endg in j_data, "endglobals anchor not found"
j_data = j_data.replace(old_endg, new_endg, 1)
print("Patch RL-5: RL global variables OK")

# ============================================================
# Patch RL-5b: Disable mode circle camera lock
# ============================================================
old_holdcam = b"function s__ModeCircleInit___HoldCam_apply takes nothing returns nothing\nset s__ModeCircleInit___HoldCam_on=true\ncall TimerStart((NewTimerEx(0)),.03125,true,function s__ModeCircleInit___HoldCam_periodic)\nendfunction"
new_holdcam = b"function s__ModeCircleInit___HoldCam_apply takes nothing returns nothing\nendfunction"
assert old_holdcam in j_data, "HoldCam_apply not found"
j_data = j_data.replace(old_holdcam, new_holdcam, 1)
print("Patch RL-5b: Mode circle camera lock disabled OK")

# ============================================================
# Patch RL-6: RL functions (inserted before main)
# ============================================================
rl_jass = """\
function RL_SendJassState takes nothing returns nothing
local integer i=0
loop
exitwhen i>=12
if s__User_hero[(i)]!=0 then
call RLSetPDAT(i,s__User_upgradeList[9*i+0],s__User_upgradeList[9*i+1],s__User_upgradeList[9*i+2],s__User_upgradeList[9*i+3],s__User_upgradeList[9*i+4],s__User_upgradeList[9*i+5],s__User_upgradeList[9*i+6],s__User_upgradeList[9*i+7],s__User_upgradeList[9*i+8],s__User_cSCooldown[i],B2I(s__User_firstSealActive[i]),s__User_firstSealCooldown[i],s__User_attributeCount[i])
endif
set i=i+1
endloop
call RLSetScore(s__Team_score[1],s__Team_score[2])
endfunction
function RL_Periodic takes nothing returns nothing
call RL_SendJassState()
call RLTick(0)
endfunction
function RL_WaitReady takes nothing returns nothing
local integer i=0
local integer count=0
local integer h
local integer idx
local integer j
local integer tmp
loop
exitwhen i>=12
if s__User_hero[(i)]!=0 then
set count=count+1
endif
set i=i+1
endloop
if count<12 then
return
endif
call DestroyTimer(GetExpiredTimer())
set rl_team1_side=RLRand(0,1)
set rl_perm[0]=0
set rl_perm[1]=1
set rl_perm[2]=2
set rl_perm[3]=3
set rl_perm[4]=4
set rl_perm[5]=5
set i=5
loop
exitwhen i<=0
set j=RLRand(0,i)
set tmp=rl_perm[i]
set rl_perm[i]=rl_perm[j]
set rl_perm[j]=tmp
set i=i-1
endloop
set i=0
loop
exitwhen i>=12
set h=s__User_hero[(i)]
if h!=0 then
set idx=i
if i>=6 then
set idx=i-6
endif
if i<6 and rl_team1_side==0 then
call SetUnitPosition(UnitIndexer___e[(h)],rl_spawn_lx[rl_perm[idx]],rl_spawn_ly[rl_perm[idx]])
elseif i<6 and rl_team1_side==1 then
call SetUnitPosition(UnitIndexer___e[(h)],rl_spawn_rx[rl_perm[idx]],rl_spawn_ry[rl_perm[idx]])
elseif i>=6 and rl_team1_side==0 then
call SetUnitPosition(UnitIndexer___e[(h)],rl_spawn_rx[rl_perm[idx]],rl_spawn_ry[rl_perm[idx]])
else
call SetUnitPosition(UnitIndexer___e[(h)],rl_spawn_lx[rl_perm[idx]],rl_spawn_ly[rl_perm[idx]])
endif
endif
set i=i+1
endloop
call ResetToGameCamera(0)
if s__User_hero[(0)]!=0 then
call SetCameraPosition(GetUnitX(UnitIndexer___e[(s__User_hero[(0)])]),GetUnitY(UnitIndexer___e[(s__User_hero[(0)])]))
else
call SetCameraPosition(0,2000)
endif
call DisplayTimedTextToPlayer(GetLocalPlayer(),0,0,10,"[RL] Teams positioned! Side="+I2S(rl_team1_side))
call JNWriteLog("[RL] ConnState="+I2S(JNGetConnectionState())+" SyncDelay="+I2S(JNGetSyncDelay()))
call DisplayTimedTextToPlayer(GetLocalPlayer(),0,0,15,"[RL] ConnState="+I2S(JNGetConnectionState())+" SyncDelay="+I2S(JNGetSyncDelay()))
call SetMapFlag(MAP_LOCK_SPEED, false)
call SetGameSpeed(MAP_SPEED_FASTEST)
call JNSetSyncDelay(10)
call JNWriteLog("[RL] After: SyncDelay="+I2S(JNGetSyncDelay()))
call DisplayTimedTextToPlayer(GetLocalPlayer(),0,0,15,"[RL] Speed=FASTEST SyncDelay->"+I2S(JNGetSyncDelay()))
call Preloader("RL_INIT")
call TimerStart(CreateTimer(),0.1,true,function RL_Periodic)
endfunction
function RL_ForceStart takes nothing returns nothing
call DestroyTimer(GetExpiredTimer())
call ModeCircleInit___GameStart(null)
call DisplayTimedTextToPlayer(GetLocalPlayer(),0,0,10,"[RL] GameStart fired - waiting for heroes...")
endfunction
function RL_Init takes nothing returns nothing
call Preloader("RL_PING")
set Playing[0]=true
set Playing[1]=true
set Playing[2]=true
set Playing[3]=true
set Playing[4]=true
set Playing[5]=true
set Playing[6]=true
set Playing[7]=true
set Playing[8]=true
set Playing[9]=true
set Playing[10]=true
set Playing[11]=true
set s__System_mode=ARENA
set s__System_selectMode=RANDOM
set s__Game_targetScore=70
set s__System_randomTeam=false
set s__System_observerOn=false
set s__System_practiceOn=false
call Fog(false)
call FogEnable(false)
call FogMaskEnable(false)
set rl_spawn_lx[0]=-6888.0
set rl_spawn_ly[0]=5085.0
set rl_spawn_lx[1]=-7106.0
set rl_spawn_ly[1]=2108.0
set rl_spawn_lx[2]=-6249.0
set rl_spawn_ly[2]=-1331.0
set rl_spawn_lx[3]=124.0
set rl_spawn_ly[3]=-101.0
set rl_spawn_lx[4]=-1408.0
set rl_spawn_ly[4]=3932.0
set rl_spawn_lx[5]=638.0
set rl_spawn_ly[5]=4757.0
set rl_spawn_rx[0]=6782.0
set rl_spawn_ry[0]=4034.0
set rl_spawn_rx[1]=7961.0
set rl_spawn_ry[1]=2360.0
set rl_spawn_rx[2]=8070.0
set rl_spawn_ry[2]=769.0
set rl_spawn_rx[3]=3437.0
set rl_spawn_ry[3]=-1280.0
set rl_spawn_rx[4]=2444.0
set rl_spawn_ry[4]=3839.0
set rl_spawn_rx[5]=8072.0
set rl_spawn_ry[5]=1837.0
call DisplayTimedTextToPlayer(GetLocalPlayer(),0,0,10,"[RL] ARENA/RANDOM/70kill 12P - event-driven init")
call TimerStart(CreateTimer(),0.1,false,function RL_ForceStart)
call TimerStart(CreateTimer(),0.5,true,function RL_WaitReady)
endfunction
""".encode('utf-8')
old_main_fn = b"function main takes nothing returns nothing"
assert old_main_fn in j_data, "main function not found"
j_data = j_data.replace(old_main_fn, rl_jass + old_main_fn, 1)
print("Patch RL-6: RL functions inserted before main OK")

# ============================================================
# Patch RL-7: Hook RL_Init in main function
# ============================================================
old_initstructs = b'call ExecuteFunc("jasshelper__initstructs531790546")'
new_initstructs = b"""call ExecuteFunc("jasshelper__initstructs531790546")
call DisplayTimedTextToPlayer(GetLocalPlayer(),0,0,30,"[RL] main reached hook point")
call RL_Init()"""
assert old_initstructs in j_data, "jasshelper initstructs not found"
j_data = j_data.replace(old_initstructs, new_initstructs, 1)
print("Patch RL-7: RL_Init hook in main OK")

# ============================================================
# Patch RL-8: Hook hero creation to set rl_hero1/rl_hero2
# ============================================================
# s__User_initialize is called when a hero is created.
# Right after it, we set rl_hero1/rl_hero2 directly in the trigger context.
hero_hook_old = b'call s__User_initialize(s__Hero_user[this],this,random)\ncall sc__HeroPool_remove(s__Hero_unitId[this])'
hero_hook_new = b"""call s__User_initialize(s__Hero_user[this],this,random)
set rl_h1_hid=GetHandleId(UnitIndexer___e[this])
call Preloader("RL_HREG|"+I2S(s__Hero_user[this])+"|"+I2S(rl_h1_hid))
call Preloader("RL_GREG|"+I2S(s__Hero_user[this])+"|"+I2S(GetHandleId(UnitIndexer___e[s__User_grail[s__Hero_user[this]]])))
call sc__HeroPool_remove(s__Hero_unitId[this])"""
count_hero = j_data.count(hero_hook_old)
assert count_hero > 0, "Hero creation hook anchor not found"
j_data = j_data.replace(hero_hook_old, hero_hook_new)
print(f"Patch RL-8: Hero creation hook OK ({count_hero} locations)")

# ============================================================
# Patch RL-9: KILL event - s__KDASystem_onDeath
# ============================================================
old_kill_anchor = b"call sc__Hero__set_killCount(killer,(s__User_score[s__Hero_user[(killer)]])+1)"
new_kill_anchor = b"call sc__Hero__set_killCount(killer,(s__User_score[s__Hero_user[(killer)]])+1)\ncall Preloader(\"RL_KILL|\"+I2S(s__Hero_user[killer])+\"|\"+I2S(s__Hero_user[hero]))"
assert old_kill_anchor in j_data, "RL_KILL: sc__Hero__set_killCount pattern not found"
j_data = j_data.replace(old_kill_anchor, new_kill_anchor, 1)
print("Patch RL-9: RL_KILL event hook OK")

# ============================================================
# Patch RL-10: CREEP event - s__Creep_onDeath
# ============================================================
# Creep_onDeath returns boolean and has locals: this, lv, dur
# Insert Preloader call after locals, before first if
old_creep_anchor = b"local real dur\nif s__Creep_respawn[this]then"
new_creep_anchor = b"local real dur\ncall Preloader(\"RL_CREEP|\"+I2S(GetPlayerId(GetOwningPlayer(GetKillingUnit()))))\nif s__Creep_respawn[this]then"
assert old_creep_anchor in j_data, "RL_CREEP: s__Creep_onDeath local/if pattern not found"
j_data = j_data.replace(old_creep_anchor, new_creep_anchor, 1)
print("Patch RL-10: RL_CREEP event hook OK")

# ============================================================
# Patch RL-11: LVUP event - s__Hero_onLevel
# ============================================================
old_lvup_anchor = b"set s__Hero_level[this]=newLevel"
new_lvup_anchor = b"set s__Hero_level[this]=newLevel\ncall Preloader(\"RL_LVUP|\"+I2S(s__Hero_user[this])+\"|\"+I2S(newLevel))"
assert old_lvup_anchor in j_data, "RL_LVUP: s__Hero_level[this]=newLevel not found"
j_data = j_data.replace(old_lvup_anchor, new_lvup_anchor, 1)
print("Patch RL-11: RL_LVUP event hook OK")

# ============================================================
# Patch RL-12: ALARM event - s__Hero_onEnter (enemy detection)
# ============================================================
# Korean text: "주변에서 서번트의 기척이 감지되었다." (actual text in JASS)
alarm_pattern = "기척이 감지되었다.".encode('utf-8')
alarm_idx = j_data.find(alarm_pattern)
assert alarm_idx >= 0, "RL_ALARM: servant detection text not found"
# Find the end of the line containing this text (after the closing paren and quote)
alarm_line_end = j_data.index(b"\n", alarm_idx)
# In s__Hero_onEnter, 'this' is the hero unit index, s__Hero_user[this] = player ID
alarm_insert = b"\ncall Preloader(\"RL_ALARM|\"+I2S(s__Hero_user[this]))"
j_data = j_data[:alarm_line_end] + alarm_insert + j_data[alarm_line_end:]
print("Patch RL-12: RL_ALARM event hook OK")

# ============================================================
# Patch RL-13: Remove PauseGame from s__Game_endExpiration
# When score=70, game should NOT freeze. C# detects via RL_SCORE.
# ============================================================
old_end_expiration = b"function s__Game_endExpiration takes nothing returns nothing\ncall ReleaseTimer(GetExpiredTimer())\ncall PauseGame(true)\ncall CinematicFadeBJ(bj_CINEFADETYPE_FADEOUT,2,\"war3mapPreview.tga\",100,100,100,20)\nendfunction"
new_end_expiration = b"function s__Game_endExpiration takes nothing returns nothing\ncall ReleaseTimer(GetExpiredTimer())\nendfunction"
assert old_end_expiration in j_data, "s__Game_endExpiration function not found"
j_data = j_data.replace(old_end_expiration, new_end_expiration, 1)
print("Patch RL-13: s__Game_endExpiration PauseGame+CinematicFade removed OK")

# ============================================================
# Patch RL-14: Add RL_DONE event to s__Game_end
# So C# can detect score-based game end immediately
# ============================================================
old_game_end = "call Text((s__Team_n[(winTeam)])+\"|c00d5aab9이 가장 먼저 점수를 획득하였으므로 게임에서 승리하였습니다.\",10)".encode('utf-8')
new_game_end = "call Text((s__Team_n[(winTeam)])+\"|c00d5aab9이 가장 먼저 점수를 획득하였으므로 게임에서 승리하였습니다.\",10)\ncall Preloader(\"RL_DONE|\"+I2S(winTeam))".encode('utf-8')
assert old_game_end in j_data, "s__Game_end victory text not found"
j_data = j_data.replace(old_game_end, new_game_end, 1)
print("Patch RL-14: s__Game_end RL_DONE event added OK")

# (Patch RL-15 removed — string display functions are NOT the crash cause)

# ============================================================
# w3i patch - map name
# ============================================================
def patch_w3i_strings(data, new_name=None, new_desc=None):
    """w3i 파일의 맵 이름/설명 문자열을 교체"""
    buf = bytearray(data)
    pos = 12  # skip format(4) + save_count(4) + editor_version(4)
    # map_name
    name_start = pos
    name_end = buf.index(0, pos)
    pos = name_end + 1
    # map_author
    pos = buf.index(0, pos) + 1
    # map_desc
    desc_start = pos
    desc_end = buf.index(0, pos)

    result = bytearray()
    result.extend(buf[:name_start])
    if new_name:
        result.extend(new_name.encode("utf-8"))
    else:
        result.extend(buf[name_start:name_end])
    result.append(0)
    result.extend(buf[name_end+1:desc_start])
    result.extend(buf[desc_start:desc_end])
    result.append(0)
    result.extend(buf[desc_end+1:])
    return bytes(result)

def patch_w3i_players_computer(data):
    """w3i 플레이어 1-11을 Computer(2)로 변경 → 로비 자동 채움"""
    buf = bytearray(data)
    off = 12  # skip format(4) + save_count(4) + editor_version(4)
    # Skip null-terminated strings: name, author, desc, recommend
    for _ in range(4):
        off = buf.index(0, off) + 1
    off += 32 + 16 + 8 + 4 + 1 + 4  # cam bounds, complements, playable, flags, tileset, LS bg
    # Skip LS strings: model, text, title, subtitle
    for _ in range(4):
        off = buf.index(0, off) + 1
    off += 4  # game data set
    # Skip prologue strings: model, text, title, subtitle
    for _ in range(4):
        off = buf.index(0, off) + 1
    # Fog settings (format version 25)
    off += 4 + 12 + 4  # fog type + start/end/density + color
    off += 4  # weather
    off = buf.index(0, off) + 1  # sound env string
    off += 1 + 4  # tileset light + water color

    num_players = struct.unpack_from('<I', buf, off)[0]
    off += 4
    patched = 0
    for i in range(num_players):
        p_num = struct.unpack_from('<I', buf, off)[0]; off += 4
        type_off = off
        p_type = struct.unpack_from('<I', buf, off)[0]; off += 4
        off += 4 + 4  # race, fixed start
        off = buf.index(0, off) + 1  # name string
        off += 4 + 4 + 4 + 4  # x, y, ally_lo, ally_hi
        if p_num > 0 and p_type == 1:  # Human -> Computer
            struct.pack_into('<I', buf, type_off, 2)
            patched += 1
    return bytes(buf), patched

w3i_data = patch_w3i_strings(orig_w3i, new_name="|c000080ffFate/Another FS 2.6RL")
print("Patch w3i: map name -> 2.6C RL OK")
w3i_data, w3i_patched = patch_w3i_players_computer(w3i_data)
print(f"Patch w3i: {w3i_patched} players -> Computer (로비 자동 채움)")

# Use original data for other files
misc_data = orig_misc
w3u_data = orig_w3u
w3t_data = orig_w3t

# ============================================================
# war3mapSkin.txt — Override loading screen "Press any key" text
# Setting LOADING_PRESS_A_KEY to empty skips the loading wait screen
# ============================================================
skin_data = b"[FrameDef]\r\nLOADING_PRESS_A_KEY=\r\n"

# ============================================================
# Save to fateanother_rl.w3x
# ============================================================
out = r".\fateanother_rl.w3x"
tmp_files = {
    r".\tmps\_t.j": (j_data, b"war3map.j"),
    r".\tmps\_t.txt": (misc_data, b"war3mapMisc.txt"),
    r".\tmps\_t.w3u": (w3u_data, b"war3map.w3u"),
    r".\tmps\_t.w3a": (w3a_data, b"war3map.w3a"),
    r".\tmps\_t.w3t": (w3t_data, b"war3map.w3t"),
    r".\tmps\_t.w3i": (w3i_data, b"war3map.w3i"),
    r".\tmps\_t.skin": (skin_data, b"war3mapSkin.txt"),
}

# Create tmps directory if not exists
os.makedirs(r".\tmps", exist_ok=True)

for fn, (data, _) in tmp_files.items():
    with open(fn, "wb") as f:
        f.write(data)

shutil.copy2(base_path, out)
hO = ctypes.c_void_p()
dll.SFileOpenArchive(out, 0, 0, ctypes.byref(hO))
flags = MPQ_FILE_COMPRESS | MPQ_FILE_REPLACEEXISTING
for fn, (_, mpqn) in tmp_files.items():
    dll.SFileAddFileEx(hO, fn, mpqn, flags, MPQ_COMPRESSION_ZLIB, MPQ_COMPRESSION_ZLIB)
dll.SFileCloseArchive(hO)

for fn in tmp_files:
    os.remove(fn)

# W3X file header patch
with open(out, "rb") as f:
    raw = f.read()
old_hdr = b"|c000080ffFate/Another FS 2.6A"
new_hdr = b"|c000080ffFate/Another FS 2.6RL"
# Pad to same length if needed
if len(new_hdr) < len(old_hdr):
    new_hdr = new_hdr + b' ' * (len(old_hdr) - len(new_hdr))
elif len(new_hdr) > len(old_hdr):
    # Truncate if too long
    new_hdr = new_hdr[:len(old_hdr)]

assert old_hdr in raw[:512], "W3X header map name not found"
raw = raw[:512].replace(old_hdr, new_hdr, 1) + raw[512:]
with open(out, "wb") as f:
    f.write(raw)
print("Patch w3x header: 2.6A -> 2.6RL OK")

# Copy to WC3 directory for -loadfile auto-start
#wc3_map = os.path.join(os.path.dirname(__file__), r"War3Client\Maps\rl\fateanother_rl.w3x")
#os.makedirs(os.path.dirname(wc3_map), exist_ok=True)
#shutil.copy2(out, wc3_map)
#print(f"Copy to WC3: {wc3_map}")

print(f"\nOutput: {out} ({os.path.getsize(out):,} bytes)")
print(f"WC3 auto-start: JNLoader -loadfile \"Maps\\rl\\fateanother_rl.w3x\" -window")
print("RL patches applied: Bug fixes + Auto hero selection + RL training env (JassNative TCP)")

#import rl_launcher
#rl_launcher.main()