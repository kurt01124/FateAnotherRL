using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Runtime.InteropServices;
using System.Linq;
using System.IO;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using Cirnix.JassNative.Runtime.Plugin;
using Cirnix.JassNative.JassAPI;
using Cirnix.JassNative.Runtime.Utilities;
using Cirnix.JassNative.Runtime.Windows;

namespace RLComm
{
    [Requires(typeof(JassAPIPlugin))]
    public class RLCommPlugin : IPlugin
    {
        // ============================================================
        // Constants
        // ============================================================
        private const int MAX_PLAYERS = 12;
        private const float TICK_INTERVAL = 0.1f;
        private const float MAP_MIN_X = -8416f;
        private const float MAP_MIN_Y = -2592f;
        private const float MAP_MAX_X = 8320f;
        private const float MAP_MAX_Y = 6176f;
        private const int GRID_W = 48;
        private const int GRID_H = 25;
        private const float CELL_SIZE = 350f;
        private const int TARGET_SCORE = 70;
        private const int HELLO_VERSION = 5;
        private const int FAIRE_TRANSFER_AMOUNT = 500;
        private const int FAIRE_CAP = 18000;

        // ---- Local Player Slot Override ----
        // Set to desired player slot (0-11) to play as that hero, or -1 to disable
        // Player 7 = FakeAssassin (H005), Player 0 = Saber (H000), etc.
        private const int LOCAL_PLAYER_SLOT_OVERRIDE = -1;
        // Game.dll CGameWar3 global pointer offset from Game.dll base
        // DAT_6fd305e0 with base 0x6f000000 → offset 0xd305e0
        private const int GAME_WAR3_PTR_OFFSET = 0xd305e0;
        // localPlayerId field offset within CGameWar3 struct
        private const int LOCAL_PLAYER_ID_OFFSET = 0x28;
        // backup field (used by replay restore)
        private const int LOCAL_PLAYER_ID_BACKUP_OFFSET = 0x2a;
        private static readonly int[] STAT_UPGRADE_CAPS = { 50, 50, 50, 50, 50, 5, 50, 50, 50 };

        // ============================================================
        // UDP Infrastructure
        // ============================================================
        private static UdpClient udpSend;
        private static UdpClient udpRecv;
        private static IPEndPoint inferenceEndpoint;
        private static int sendPort = 7777;  // C++ server port
        private static int recvPort = 7778;  // Our receive port for ACTION packets
        private static volatile bool running;

        // ============================================================
        // Hero State
        // ============================================================
        private static JassUnit[] heroes = new JassUnit[MAX_PLAYERS];
        private static int[] heroHandleIds = new int[MAX_PLAYERS];
        private static int[] heroTypeIds = new int[MAX_PLAYERS];  // original typeId at registration (morph-proof)
        private static bool[] heroRegistered = new bool[MAX_PLAYERS];
        private static int heroCount;
        private static volatile bool heroesReady;

        private static int tickCount;
        private static int episodeState; // 0=running, 1=team0_wipe, 2=team1_wipe, 3=timeout, 4=score_reached

        // Targeting skill grace period: suppress move commands for N ticks after issuing a targeting skill
        // to prevent immediate cancellation by next tick's move order
        private static int[] _targetSkillGraceTick = new int[MAX_PLAYERS];
        private const int TARGETING_SKILL_GRACE = 50; // ~0.5s at 10x (100 ticks/s)

        // Previous position for velocity computation
        private static float[] _prevX = new float[MAX_PLAYERS];
        private static float[] _prevY = new float[MAX_PLAYERS];
        private static bool _prevPosValid;

        // Wall-clock tick timing diagnostics
        private static uint _firstTickWallMs;
        private static uint _lastTickLogWallMs;
        private static bool _firstTickWallSet;

        // Pathability grid (cached once on first tick)
        private static int[] _pathabilityGrid;
        private static bool _pathabilityComputed;

        // Creep tracking
        private const int MAX_CREEPS = 70;
        private static readonly int CREEP_TYPE_H043 = FourCC("h043");
        private static readonly int CREEP_TYPE_H044 = FourCC("h044");
        private static readonly int CREEP_TYPE_H045 = FourCC("h045");
        private static JassGroup _creepGroup;  // reusable group for enumeration

        // ============================================================
        // Reusable static buffers (avoid per-tick allocations)
        // ============================================================
        private static int[] _visGridTeam0 = new int[GRID_W * GRID_H];
        private static int[] _visGridTeam1 = new int[GRID_W * GRID_H];
        private static float[] _curX = new float[MAX_PLAYERS];
        private static float[] _curY = new float[MAX_PLAYERS];
        private static JassPlayer[] _playersCache = new JassPlayer[MAX_PLAYERS];
        // Reusable MemoryStream to avoid per-tick allocation
        private static MemoryStream _stateMs = new MemoryStream(8192);
        private static BinaryWriter _stateWriter = new BinaryWriter(_stateMs);
        private static byte[] _stateBuffer = new byte[8192];

        // ============================================================
        // ActionMaskBits -- zero-allocation replacement for JObject masks
        // ============================================================
        private struct ActionMaskBits
        {
            public byte skill;          // 8 bits (indices 0-7)
            public ushort unitTarget;   // 16 bits (indices 0-13, upper 2 unused)
            public byte skillLevelup;   // 8 bits (indices 0-5, upper 2 unused)
            public ushort statUpgrade;  // 16 bits (indices 0-9, upper 6 unused)
            public byte attribute;      // 8 bits (indices 0-4, upper 3 unused)
            public uint itemBuy;        // 32 bits (indices 0-16, 17 total)
            public byte itemUse;        // 8 bits (indices 0-6, upper 1 unused)
            public byte sealUse;        // 8 bits (indices 0-6, upper 1 unused)
            public byte faireSend;      // 8 bits (indices 0-5, upper 2 unused)
            public byte faireRequest;   // 8 bits (indices 0-5, upper 2 unused)
            public byte faireRespond;   // 8 bits (indices 0-2, upper 5 unused)
        }

        // Faire (기사회생) tracking -- per-player PVP death count and usage
        private static int[] _advCount = new int[MAX_PLAYERS];   // PVP deaths (enemy kills only, not creep/suicide)
        private static int[] _faireUsed = new int[MAX_PLAYERS];  // how many faires consumed

        // ============================================================
        // JassStateCache -- data from JASS Preloader messages
        // ============================================================
        private static class JassStateCache
        {
            public static int[,] upgrades = new int[MAX_PLAYERS, 9]; // [pid, upgradeId 0-8]
            public static int[] sealCd = new int[MAX_PLAYERS];
            public static int[] firstActive = new int[MAX_PLAYERS]; // 0 or 1
            public static int[] firstRemain = new int[MAX_PLAYERS];
            public static int[] attrCount = new int[MAX_PLAYERS];
            public static int[,] attrAcquired = new int[MAX_PLAYERS, 5]; // [pid, 1-4] per-attribute acquire count
            public static int[] teamScore = new int[2]; // [team0, team1]

            public static void Reset()
            {
                upgrades = new int[MAX_PLAYERS, 9];
                sealCd = new int[MAX_PLAYERS];
                firstActive = new int[MAX_PLAYERS];
                firstRemain = new int[MAX_PLAYERS];
                attrCount = new int[MAX_PLAYERS];
                attrAcquired = new int[MAX_PLAYERS, 5];
                teamScore = new int[2];
            }
        }

        // Grail units
        private static JassUnit[] _grailUnits = new JassUnit[MAX_PLAYERS];
        private static bool[] _grailRegistered = new bool[MAX_PLAYERS];

        // ============================================================
        // CooldownTracker
        // ============================================================
        private struct CdEntry { public float useTime; public float maxCd; }
        private struct SealResult { public JassItem item; public int charges; }
        private struct FaireRequest { public int requester; public int amount; }

        private class CooldownTracker
        {
            private Dictionary<long, CdEntry> _tracker = new Dictionary<long, CdEntry>();

            private static long Key(int heroIdx, int slotIdx) { return heroIdx * 10L + slotIdx; }

            public void OnSkillUsed(int heroIdx, int slotIdx, float gameTime, float maxCd)
            {
                _tracker[Key(heroIdx, slotIdx)] = new CdEntry { useTime = gameTime, maxCd = maxCd };
            }

            public float GetCdRemain(int heroIdx, int slotIdx, float gameTime)
            {
                long key = Key(heroIdx, slotIdx);
                if (!_tracker.TryGetValue(key, out var entry)) return 0f;
                float remain = entry.maxCd - (gameTime - entry.useTime);
                return remain > 0f ? remain : 0f;
            }

            public void ResetAll(int heroIdx)
            {
                var keysToRemove = new List<long>();
                foreach (var k in _tracker.Keys)
                    if (k / 10 == heroIdx) keysToRemove.Add(k);
                foreach (var k in keysToRemove)
                    _tracker.Remove(k);
            }

            public void Clear()
            {
                _tracker.Clear();
            }
        }
        private static CooldownTracker _cdTracker = new CooldownTracker();
        private static int[] _maskDebugCount = new int[MAX_PLAYERS];

        // ============================================================
        // EventQueue (binary-friendly)
        // ============================================================
        private struct BinaryEvent
        {
            public byte type;       // 1=KILL, 2=CREEP_KILL, 3=LEVEL_UP
            public byte killerIdx;  // or unit_idx for LEVEL_UP
            public byte victimIdx;  // or new_level for LEVEL_UP
            public byte padding;
            public uint tick;
        }

        private class EventQueue
        {
            private List<BinaryEvent> _queue = new List<BinaryEvent>();

            public void AddBinary(BinaryEvent evt) { _queue.Add(evt); }

            // Legacy JObject support for existing Preloader code
            public void Add(JObject evt)
            {
                string evtType = evt.Value<string>("type");
                BinaryEvent be = new BinaryEvent();
                be.tick = (uint)(evt.Value<int>("tick"));

                if (evtType == "KILL")
                {
                    be.type = 1;
                    be.killerIdx = (byte)evt.Value<int>("killer");
                    be.victimIdx = (byte)evt.Value<int>("victim");
                }
                else if (evtType == "CREEP_KILL")
                {
                    be.type = 2;
                    be.killerIdx = (byte)evt.Value<int>("killer");
                    be.victimIdx = 0;
                }
                else if (evtType == "LEVEL_UP")
                {
                    be.type = 3;
                    be.killerIdx = (byte)evt.Value<int>("unit_idx");
                    be.victimIdx = (byte)evt.Value<int>("new_level");
                }
                else if (evtType == "PORTAL")
                {
                    be.type = 4;
                    be.killerIdx = (byte)evt.Value<int>("unit_idx");
                    be.victimIdx = 0;
                }

                _queue.Add(be);
            }

            public List<BinaryEvent> FlushAndClearBinary()
            {
                var copy = new List<BinaryEvent>(_queue);
                _queue.Clear();
                return copy;
            }

            public JArray FlushAndClear()
            {
                // Keep for backward compat if needed
                var arr = new JArray();
                _queue.Clear();
                return arr;
            }

            public void Clear() { _queue.Clear(); }
        }
        private static EventQueue _eventQueue = new EventQueue();

        // ============================================================
        // Alarm State
        // ============================================================
        private static bool[] _alarmState = new bool[MAX_PLAYERS];

        // ============================================================
        // Faire Request Manager
        // ============================================================
        // Key = target player idx, Value = FaireRequest
        private static Dictionary<int, FaireRequest> _faireRequests = new Dictionary<int, FaireRequest>();

        // ============================================================
        // Runtime RNG (seeded by Environment.TickCount)
        // ============================================================
        private static readonly Random _rng = new Random(Environment.TickCount);

        private delegate JassInteger RLRandDelegate(JassInteger min, JassInteger max);
        private static JassInteger RLRand(JassInteger min, JassInteger max)
        {
            return _rng.Next((int)min, (int)max + 1);
        }

        // ============================================================
        // Integer-based JASS→C# data natives (zero string handles)
        // ============================================================
        private delegate void RLSetPDATDelegate(
            JassInteger pid,
            JassInteger u0, JassInteger u1, JassInteger u2,
            JassInteger u3, JassInteger u4, JassInteger u5,
            JassInteger u6, JassInteger u7, JassInteger u8,
            JassInteger sealCd, JassInteger sealActive,
            JassInteger sealFirstCd, JassInteger attrCount);
        private static void RLSetPDAT(
            JassInteger pid,
            JassInteger u0, JassInteger u1, JassInteger u2,
            JassInteger u3, JassInteger u4, JassInteger u5,
            JassInteger u6, JassInteger u7, JassInteger u8,
            JassInteger sealCd, JassInteger sealActive,
            JassInteger sealFirstCd, JassInteger attrCount)
        {
            int p = (int)pid;
            if (p < 0 || p >= MAX_PLAYERS) return;
            JassStateCache.upgrades[p, 0] = (int)u0;
            JassStateCache.upgrades[p, 1] = (int)u1;
            JassStateCache.upgrades[p, 2] = (int)u2;
            JassStateCache.upgrades[p, 3] = (int)u3;
            JassStateCache.upgrades[p, 4] = (int)u4;
            JassStateCache.upgrades[p, 5] = (int)u5;
            JassStateCache.upgrades[p, 6] = (int)u6;
            JassStateCache.upgrades[p, 7] = (int)u7;
            JassStateCache.upgrades[p, 8] = (int)u8;
            JassStateCache.sealCd[p] = (int)sealCd;
            JassStateCache.firstActive[p] = (int)sealActive;
            JassStateCache.firstRemain[p] = (int)sealFirstCd;
            JassStateCache.attrCount[p] = (int)attrCount;
        }

        private delegate void RLSetScoreDelegate(JassInteger t1, JassInteger t2);
        private static void RLSetScore(JassInteger t1, JassInteger t2)
        {
            JassStateCache.teamScore[0] = (int)t1;
            JassStateCache.teamScore[1] = (int)t2;
        }

        private delegate void RLTickDelegate(JassInteger tick);
        private static void RLTickNative(JassInteger tick)
        {
            OnTick();
        }

        // Kill event: killer_pid, victim_pid
        private delegate void RLKillDelegate(JassInteger killer, JassInteger victim);
        private static void RLKillNative(JassInteger killer, JassInteger victim)
        {
            int k = (int)killer, v = (int)victim;
            _eventQueue.Add(new JObject
            {
                ["type"] = "KILL",
                ["killer"] = k,
                ["victim"] = v,
                ["tick"] = tickCount
            });
            if (k >= 0 && k < MAX_PLAYERS && v >= 0 && v < MAX_PLAYERS && k != v)
            {
                int kTeam = k < 6 ? 0 : 1, vTeam = v < 6 ? 0 : 1;
                if (kTeam != vTeam)
                {
                    _advCount[v]++;
                    if (_advCount[v] % 7 == 0)
                        Log($"[RLComm] Faire available for p{v} (advCount={_advCount[v]})");
                }
            }
        }

        // Creep kill: killer_pid
        private delegate void RLCreepDelegate(JassInteger killer);
        private static void RLCreepNative(JassInteger killer)
        {
            _eventQueue.Add(new JObject
            {
                ["type"] = "CREEP_KILL",
                ["killer"] = (int)killer,
                ["tick"] = tickCount
            });
        }

        // Level up: pid, newLevel
        private delegate void RLLvUpDelegate(JassInteger pid, JassInteger level);
        private static void RLLvUpNative(JassInteger pid, JassInteger level)
        {
            _eventQueue.Add(new JObject
            {
                ["type"] = "LEVEL_UP",
                ["unit_idx"] = (int)pid,
                ["new_level"] = (int)level,
                ["tick"] = tickCount
            });
        }

        // Portal: pid
        private delegate void RLPortalDelegate(JassInteger pid);
        private static void RLPortalNative(JassInteger pid)
        {
            int p = (int)pid;
            _eventQueue.Add(new JObject
            {
                ["type"] = "PORTAL",
                ["unit_idx"] = p,
                ["tick"] = tickCount
            });
        }

        // Alarm: pid
        private delegate void RLAlarmDelegate(JassInteger pid);
        private static void RLAlarmNative(JassInteger pid)
        {
            int p = (int)pid;
            if (p >= 0 && p < MAX_PLAYERS)
                _alarmState[p] = true;
        }

        // Game done: winTeam
        private delegate void RLDoneDelegate(JassInteger winTeam);
        private static void RLDoneNative(JassInteger winTeam)
        {
            int wt = (int)winTeam;
            Log($"[RLComm] RL_DONE native: winTeam={wt}, triggering episode end");
            episodeState = 4;
        }

        private delegate JassInteger GetRandomIntDelegate(JassInteger low, JassInteger high);
        private static JassInteger GetRandomIntOverride(JassInteger low, JassInteger high)
        {
            int lo = (int)low, hi = (int)high;
            if (lo > hi) { int t = lo; lo = hi; hi = t; }
            return _rng.Next(lo, hi + 1);
        }

        private delegate JassRealRet GetRandomRealDelegate(JassRealArg low, JassRealArg high);
        private static JassRealRet GetRandomRealOverride(JassRealArg low, JassRealArg high)
        {
            float lo = (float)low, hi = (float)high;
            if (lo > hi) { float t = lo; lo = hi; hi = t; }
            return (float)(lo + _rng.NextDouble() * (hi - lo));
        }

        // ============================================================
        // Speed Acceleration -- LOOP Tick Hook
        // ============================================================
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate uint LoopTickDelegate();

        private static LoopTickDelegate _origLoopTick;
        private static uint _tickBase;
        private static bool _tickBaseSet;

        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        private delegate IntPtr SetTimerDelegate(IntPtr hWnd, IntPtr nIDEvent, uint uElapse, IntPtr lpTimerFunc);

        private static SetTimerDelegate _origSetTimer;
        private static double _speedMultiplier = 1.0;
        private static long _cntSetTimer;
        private static long _cntLoopTick;

        [DllImport("winmm.dll")]
        private static extern uint timeBeginPeriod(uint uPeriod);

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool VirtualProtect(IntPtr lpAddress, uint dwSize, uint flNewProtect, out uint lpflOldProtect);

        [DllImport("kernel32.dll")]
        private static extern int VirtualQuery(IntPtr lpAddress, out MEMORY_BASIC_INFORMATION lpBuffer, int dwLength);

        [StructLayout(LayoutKind.Sequential)]
        private struct MEMORY_BASIC_INFORMATION
        {
            public IntPtr BaseAddress;
            public IntPtr AllocationBase;
            public uint AllocationProtect;
            public IntPtr RegionSize;
            public uint State;
            public uint Protect;
            public uint Type;
        }

        // Direct kernel32 import -- always returns real wall-clock time
        [DllImport("kernel32.dll", EntryPoint = "GetTickCount")]
        private static extern uint RealGetTickCount();

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern void ExitProcess(uint uExitCode);

        [DllImport("user32.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr FindWindowA(string lpClassName, string lpWindowName);

        [DllImport("user32.dll")]
        private static extern bool PostMessageA(IntPtr hWnd, uint Msg, IntPtr wParam, IntPtr lParam);

        [DllImport("user32.dll")]
        private static extern bool SetForegroundWindow(IntPtr hWnd);

        [DllImport("user32.dll")]
        private static extern void keybd_event(byte bVk, byte bScan, uint dwFlags, UIntPtr dwExtraInfo);

        private const uint WM_KEYDOWN = 0x100;
        private const uint WM_KEYUP = 0x101;

        private const uint PAGE_EXECUTE_READWRITE = 0x40;
        private const uint MEM_COMMIT = 0x1000;
        // Writable page protection mask: RW(0x04) | WC(0x08) | ERW(0x40) | EWC(0x80)
        private const uint WRITABLE_MASK = 0xCC;
        private const uint PAGE_GUARD = 0x100;

        // Turn interval memory scanner state
        private static readonly List<IntPtr> _scanCandidates = new List<IntPtr>();
        private static readonly List<IntPtr> _turnIntervalAddrs = new List<IntPtr>();

        private static IntPtr SetTimerHook(IntPtr hWnd, IntPtr nIDEvent, uint uElapse, IntPtr lpTimerFunc)
        {
            Interlocked.Increment(ref _cntSetTimer);
            uint adjusted = (uint)(uElapse / _speedMultiplier);
            if (adjusted == 0) adjusted = 1;
            if (_cntSetTimer <= 10) // only log first 10 to reduce spam
                Log($"[RLComm] SetTimer: id={nIDEvent}, orig={uElapse}ms, adj={adjusted}ms");
            return _origSetTimer(hWnd, nIDEvent, adjusted, lpTimerFunc);
        }

        /// <summary>
        /// LOOP tick hook -- accelerates the game loop's sense of time.
        /// Returns baseTick + (realTick - baseTick) * multiplier.
        /// </summary>
        private static uint LoopTickHook()
        {
            Interlocked.Increment(ref _cntLoopTick);
            uint realTick = _origLoopTick();
            if (!_tickBaseSet)
            {
                _tickBase = realTick;
                _tickBaseSet = true;
                return realTick;
            }
            uint elapsed = realTick - _tickBase;
            return _tickBase + (uint)(elapsed * _speedMultiplier);
        }

        /// <summary>
        /// Phase 1: Scan ALL writable committed memory for DWORD = 42.
        /// </summary>
        private static void ScanPhase1(IntPtr gameDll)
        {
            _scanCandidates.Clear();
            long totalScanned = 0;
            int mbiSize = Marshal.SizeOf(typeof(MEMORY_BASIC_INFORMATION));
            IntPtr addr = new IntPtr(0x10000); // skip null page
            long maxAddr = 0x7FFE0000L;
            const int MAX_CANDIDATES = 200000;

            while (addr.ToInt64() < maxAddr && _scanCandidates.Count < MAX_CANDIDATES)
            {
                MEMORY_BASIC_INFORMATION mbi;
                if (VirtualQuery(addr, out mbi, mbiSize) == 0)
                    break;

                long regionSize = mbi.RegionSize.ToInt64();
                if (regionSize <= 0) break;

                // Only scan committed, writable, non-guard pages
                if (mbi.State == MEM_COMMIT &&
                    (mbi.Protect & WRITABLE_MASK) != 0 &&
                    (mbi.Protect & PAGE_GUARD) == 0)
                {
                    long regionEnd = mbi.BaseAddress.ToInt64() + regionSize;
                    for (long p = mbi.BaseAddress.ToInt64(); p <= regionEnd - 4; p += 4)
                    {
                        try
                        {
                            int val = Marshal.ReadInt32(new IntPtr(p));
                            if (val == 42)
                            {
                                _scanCandidates.Add(new IntPtr(p));
                                if (_scanCandidates.Count >= MAX_CANDIDATES) break;
                            }
                        }
                        catch { break; }
                    }
                    totalScanned += regionSize;
                }

                addr = new IntPtr(mbi.BaseAddress.ToInt64() + regionSize);
            }

            // Classify candidates by module
            int inGameDll = 0, inOther = 0;
            long gameDllStart = gameDll.ToInt64();
            long gameDllEnd = gameDllStart + 0xE00000; // ~14MB image
            foreach (IntPtr c in _scanCandidates)
            {
                if (c.ToInt64() >= gameDllStart && c.ToInt64() < gameDllEnd) inGameDll++;
                else inOther++;
            }

            Log($"[RLComm] SCAN1: {_scanCandidates.Count} DWORD=42 in {totalScanned / 1024}KB writable mem (game.dll={inGameDll}, other={inOther})");
        }

        /// <summary>
        /// Phase 2: After SetGameSpeed(Normal), rescan candidates.
        /// </summary>
        private static void ScanPhase2()
        {
            _turnIntervalAddrs.Clear();
            IntPtr gameDll = Kernel32.GetModuleHandleA("game.dll");
            long gameDllBase = gameDll.ToInt64();
            int unchanged = 0, changed100 = 0, changedOther = 0;

            foreach (IntPtr addr in _scanCandidates)
            {
                try
                {
                    int val = Marshal.ReadInt32(addr);
                    if (val == 100)
                    {
                        _turnIntervalAddrs.Add(addr);
                        changed100++;
                        string location;
                        long diff = addr.ToInt64() - gameDllBase;
                        if (diff >= 0 && diff < 0xE00000)
                            location = $"game.dll+0x{diff:X}";
                        else
                            location = $"0x{addr.ToInt64():X8}";

                        StringBuilder ctx = new StringBuilder();
                        for (int off = -16; off <= 16; off += 4)
                        {
                            if (off == 0) ctx.Append("[");
                            try { ctx.AppendFormat("{0}", Marshal.ReadInt32(addr + off)); }
                            catch { ctx.Append("??"); }
                            if (off == 0) ctx.Append("]");
                            ctx.Append(" ");
                        }
                        Log($"[RLComm] SCAN2: MATCH {location} ctx={ctx.ToString().Trim()}");
                    }
                    else if (val == 42)
                        unchanged++;
                    else
                        changedOther++;
                }
                catch { changedOther++; }
            }

            Log($"[RLComm] SCAN2: {changed100} match(42->100), {unchanged} unchanged, {changedOther} changed-other, total={_scanCandidates.Count}");
        }

        /// <summary>
        /// Phase 3: Patch turn interval addresses to desired ms value.
        /// </summary>
        private static void PatchTurnInterval()
        {
            if (_turnIntervalAddrs.Count == 0)
            {
                Log("[RLComm] SPEEDPATCH: No turn interval addresses found!");
                Log("[RLComm] SPEEDPATCH: Run RL_SCAN1 (at Fastest), change to Normal, then RL_SCAN2 first.");
                return;
            }

            int desiredMs = Math.Max(1, (int)(42.0 / _speedMultiplier));
            IntPtr gameDll = Kernel32.GetModuleHandleA("game.dll");

            foreach (IntPtr addr in _turnIntervalAddrs)
            {
                try
                {
                    int curVal = Marshal.ReadInt32(addr);
                    uint oldProt;
                    VirtualProtect(addr, 4, PAGE_EXECUTE_READWRITE, out oldProt);
                    Marshal.WriteInt32(addr, desiredMs);
                    VirtualProtect(addr, 4, oldProt, out _);

                    long rva = addr.ToInt64() - gameDll.ToInt64();
                    Log($"[RLComm] SPEEDPATCH: game.dll+0x{rva:X} {curVal}->{desiredMs}ms");
                }
                catch (Exception ex)
                {
                    Log($"[RLComm] SPEEDPATCH error: {ex.Message}");
                }
            }

            Log($"[RLComm] SPEEDPATCH: {_turnIntervalAddrs.Count} addr(s) patched to {desiredMs}ms (target {_speedMultiplier}x)");
        }

        /// <summary>Re-apply turn interval patch periodically (game may recalculate)</summary>
        private static void ReapplySpeedPatch()
        {
            if (_turnIntervalAddrs.Count == 0 || _speedMultiplier <= 1.0) return;
            int desiredMs = Math.Max(1, (int)(42.0 / _speedMultiplier));
            foreach (IntPtr addr in _turnIntervalAddrs)
            {
                try { Marshal.WriteInt32(addr, desiredMs); } catch { }
            }
        }

        // ============================================================
        // Fog of War Bitmap Direct Read (zero JASS native calls)
        // ============================================================
        // Pointer chain: game.dll + RVA → CGameWar3* → +0x34 → CFogOfWarMap*
        // CFogOfWarMap layout (from Ghidra reverse engineering):
        //   +0x2C = REVEAL layer (bit SET = currently visible) ← THIS is what we read
        //   +0x30 = BLACK MASK layer (bit SET = never explored, black fog)
        //   +0x34 = layer2 bitmap ptr
        //   +0x38 = layer3 bitmap ptr (altitude/height-based)
        //   +0x60 = grid width (int)
        //   +0x64 = row stride (int)
        //   +0x68 = shift (int, log2 of row stride)
        //   +0x6C = grid height (int)
        // Each cell = uint16, bit N = player N
        //
        // IsVisibleToPlayer logic:
        //   blackMask(0x30) bit CLEAR AND reveal(0x2C) bit SET → VISIBLE
        //   0x30 = real-time "currently visible" layer (cleared & rebuilt every 4 ticks)
        //   0x2C = explored/revealed layer (persistent, inverted: bit=0 means explored)
        private const int RVA_CGAMEWAR3_PTR = 0xD305E0;
        private const int OFF_FOGMAP = 0x34;       // CGameWar3 → CFogOfWarMap*
        private const int OFF_FOG_VISIBLE = 0x30;  // CFogOfWarMap → real-time visibility bitmap
        private const int OFF_FOG_WIDTH = 0x60;
        private const int OFF_FOG_STRIDE = 0x64;
        private const int OFF_FOG_SHIFT = 0x68;
        private const int OFF_FOG_HEIGHT = 0x6C;

        // Cached fog pointers (initialized once)
        private static IntPtr _fogBitmapPtr;       // visible fog bitmap base address
        private static int _fogGridW, _fogGridH, _fogShift;
        private static bool _fogInitialized;
        private static bool _fogInitFailed;

        // Fog grid origin in world coordinates (computed from grid dimensions)
        private static float _fogOriginX, _fogOriginY;
        private static float _fogCellSize = 128f;  // WC3 standard terrain cell

        // Team bitmasks for fog cells: team 0 = players 0-5, team 1 = players 6-11
        private const ushort FOG_TEAM0_MASK = 0x003F;  // bits 0-5
        private const ushort FOG_TEAM1_MASK = 0x0FC0;  // bits 6-11

        /// <summary>
        /// Initialize fog bitmap pointers by following the game.dll pointer chain.
        /// Must be called after game.dll is loaded. Safe to call multiple times.
        /// </summary>
        private static bool InitFogPointers()
        {
            if (_fogInitialized) return true;
            if (_fogInitFailed) return false;

            try
            {
                IntPtr gameDll = Kernel32.GetModuleHandleA("game.dll");
                if (gameDll == IntPtr.Zero)
                {
                    Log("[RLComm] FOG: game.dll not found");
                    return false;
                }

                // Step 1: Read CGameWar3* from global pointer
                IntPtr ptrAddr = gameDll + RVA_CGAMEWAR3_PTR;
                int gameWar3Raw = Marshal.ReadInt32(ptrAddr);
                if (gameWar3Raw == 0)
                {
                    Log("[RLComm] FOG: CGameWar3 pointer is NULL");
                    return false;
                }
                IntPtr gameWar3 = new IntPtr(gameWar3Raw);

                // Step 2: Read CFogOfWarMap* from CGameWar3 + 0x34
                int fogMapRaw = Marshal.ReadInt32(gameWar3 + OFF_FOGMAP);
                if (fogMapRaw == 0)
                {
                    Log("[RLComm] FOG: CFogOfWarMap pointer is NULL");
                    return false;
                }
                IntPtr fogMap = new IntPtr(fogMapRaw);

                // Step 3: Read fog grid dimensions
                _fogGridW = Marshal.ReadInt32(fogMap + OFF_FOG_WIDTH);
                _fogGridH = Marshal.ReadInt32(fogMap + OFF_FOG_HEIGHT);
                _fogShift = Marshal.ReadInt32(fogMap + OFF_FOG_SHIFT);
                int stride = Marshal.ReadInt32(fogMap + OFF_FOG_STRIDE);

                // Step 4: Read reveal (currently visible) fog bitmap pointer
                int bitmapRaw = Marshal.ReadInt32(fogMap + OFF_FOG_VISIBLE);
                if (bitmapRaw == 0)
                {
                    Log("[RLComm] FOG: reveal bitmap pointer is NULL");
                    _fogInitFailed = true;
                    return false;
                }
                _fogBitmapPtr = new IntPtr(bitmapRaw);

                // Step 5: Compute fog grid origin from grid dimensions
                // WC3 terrain grid is centered near (0,0). The fog grid covers the entire
                // terrain area. Origin = center - half_extent.
                // Map center from known camera bounds:
                float mapCenterX = (MAP_MIN_X + MAP_MAX_X) / 2f;
                float mapCenterY = (MAP_MIN_Y + MAP_MAX_Y) / 2f;
                _fogOriginX = mapCenterX - (_fogGridW * _fogCellSize / 2f);
                _fogOriginY = mapCenterY - (_fogGridH * _fogCellSize / 2f);

                _fogInitialized = true;

                Log($"[RLComm] FOG INIT SUCCESS:");
                Log($"[RLComm]   game.dll base = 0x{gameDll.ToInt64():X8}");
                Log($"[RLComm]   CGameWar3* = 0x{gameWar3Raw:X8}");
                Log($"[RLComm]   CFogOfWarMap* = 0x{fogMapRaw:X8}");
                Log($"[RLComm]   visibility bitmap (0x30) = 0x{bitmapRaw:X8}");
                Log($"[RLComm]   grid: {_fogGridW}x{_fogGridH}, stride={stride}, shift={_fogShift}");
                Log($"[RLComm]   fogOrigin: ({_fogOriginX}, {_fogOriginY})");
                Log($"[RLComm]   fogCellSize: {_fogCellSize}");
                Log($"[RLComm]   terrain extent: ({_fogOriginX}, {_fogOriginY}) to ({_fogOriginX + _fogGridW * _fogCellSize}, {_fogOriginY + _fogGridH * _fogCellSize})");

                return true;
            }
            catch (Exception ex)
            {
                Log($"[RLComm] FOG INIT FAILED: {ex.Message}");
                _fogInitFailed = true;
                return false;
            }
        }

        /// <summary>
        /// Read a fog cell value (uint16 bitmask) at fog grid coordinates.
        /// Returns 0 if coordinates are out of bounds or read fails.
        /// </summary>
        private static ushort ReadFogCell(int fogX, int fogY)
        {
            if (fogX < 0 || fogX >= _fogGridW || fogY < 0 || fogY >= _fogGridH)
                return 0;
            int offset = ((fogY << (_fogShift & 0x1F)) + fogX) * 2;
            return (ushort)Marshal.ReadInt16(_fogBitmapPtr + offset);
        }

        /// <summary>
        /// Convert world coordinates to fog grid coordinates.
        /// </summary>
        private static void WorldToFog(float worldX, float worldY, out int fogX, out int fogY)
        {
            fogX = (int)((worldX - _fogOriginX) / _fogCellSize);
            fogY = (int)((worldY - _fogOriginY) / _fogCellSize);
        }

        /// <summary>
        /// Check if a world position is visible to a team using the fog bitmap.
        /// teamStartPid: 0 for team0, 6 for team1.
        /// </summary>
        private static bool IsPositionVisibleToTeam(float worldX, float worldY, int teamStartPid)
        {
            int fogX, fogY;
            WorldToFog(worldX, worldY, out fogX, out fogY);
            ushort cell = ReadFogCell(fogX, fogY);
            ushort teamMask = teamStartPid == 0 ? FOG_TEAM0_MASK : FOG_TEAM1_MASK;
            return (cell & teamMask) != 0;
        }

        /// <summary>
        /// Check if a unit at (worldX, worldY) is visible to a specific player.
        /// Replaces Natives.IsUnitVisible() which leaks memory on Wine.
        /// </summary>
        private static bool IsUnitVisibleViaFog(float worldX, float worldY, int playerIndex)
        {
            int fogX, fogY;
            WorldToFog(worldX, worldY, out fogX, out fogY);
            ushort cell = ReadFogCell(fogX, fogY);
            ushort playerBit = (ushort)(1 << (playerIndex & 0x1F));
            return (cell & playerBit) != 0;
        }

        /// <summary>
        /// Compute visibility grid for a team by reading fog bitmap directly.
        /// Replaces ComputeVisibilityGridInPlace (math-based circular vision).
        /// Zero JASS native calls. Perfectly accurate (accounts for walls, abilities, etc).
        /// </summary>
        private static void ComputeVisibilityFromFogBitmap(int teamStartPid, int[] grid)
        {
            Array.Clear(grid, 0, grid.Length);
            ushort teamMask = teamStartPid == 0 ? FOG_TEAM0_MASK : FOG_TEAM1_MASK;

            for (int gy = 0; gy < GRID_H; gy++)
            {
                float wy = MAP_MIN_Y + (gy + 0.5f) * CELL_SIZE;
                for (int gx = 0; gx < GRID_W; gx++)
                {
                    float wx = MAP_MIN_X + (gx + 0.5f) * CELL_SIZE;
                    int fogX, fogY;
                    WorldToFog(wx, wy, out fogX, out fogY);
                    ushort cell = ReadFogCell(fogX, fogY);
                    if ((cell & teamMask) != 0)
                        grid[gy * GRID_W + gx] = 1;
                }
            }
        }

        // Cached fog layer pointers for multi-layer reading
        private static IntPtr _fogMaskPtr;    // 0x2C - explored (mask) fog
        private static IntPtr _fogLayer3Ptr;  // 0x34
        private static IntPtr _fogLayer4Ptr;  // 0x38

        /// <summary>
        /// Diagnostic: validate fog bitmap by checking hero positions.
        /// Heroes should always be visible to their own team.
        /// </summary>
        private static void ValidateFogMapping()
        {
            if (!_fogInitialized) return;

            // Also read other layer pointers for diagnostics
            try
            {
                IntPtr gameDll = Kernel32.GetModuleHandleA("game.dll");
                IntPtr gameWar3 = new IntPtr(Marshal.ReadInt32(gameDll + RVA_CGAMEWAR3_PTR));
                IntPtr fogMap = new IntPtr(Marshal.ReadInt32(gameWar3 + OFF_FOGMAP));
                _fogMaskPtr = new IntPtr(Marshal.ReadInt32(fogMap + 0x28));   // allocation base (0x28)
                _fogLayer3Ptr = new IntPtr(Marshal.ReadInt32(fogMap + 0x34));
                _fogLayer4Ptr = new IntPtr(Marshal.ReadInt32(fogMap + 0x38));
            }
            catch { }

            var sb = new StringBuilder();
            sb.AppendFormat("[RLComm] FOG VALIDATION (tick={0}):\n", tickCount);
            int pass = 0, fail = 0;

            for (int i = 0; i < MAX_PLAYERS; i++)
            {
                if (!heroRegistered[i]) continue;
                float hp = 0f;
                try { hp = Natives.GetUnitState(heroes[i], JassUnitState.Life); } catch { }
                if (hp <= 0.405f) continue;

                float hx = 0f, hy = 0f;
                try { hx = Natives.GetUnitX(heroes[i]); } catch { }
                try { hy = Natives.GetUnitY(heroes[i]); } catch { }

                int fogX, fogY;
                WorldToFog(hx, hy, out fogX, out fogY);
                ushort cellVis = ReadFogCell(fogX, fogY);

                // Also read other layers at the same position
                ushort cellMask = 0, cellL3 = 0, cellL4 = 0;
                if (fogX >= 0 && fogX < _fogGridW && fogY >= 0 && fogY < _fogGridH)
                {
                    int offset = ((fogY << (_fogShift & 0x1F)) + fogX) * 2;
                    try { cellMask = (ushort)Marshal.ReadInt16(_fogMaskPtr + offset); } catch { }
                    try { cellL3 = (ushort)Marshal.ReadInt16(_fogLayer3Ptr + offset); } catch { }
                    try { cellL4 = (ushort)Marshal.ReadInt16(_fogLayer4Ptr + offset); } catch { }
                }

                // Hero should be visible to own team
                int ownTeamStart = i < 6 ? 0 : 6;
                ushort ownMask = ownTeamStart == 0 ? FOG_TEAM0_MASK : FOG_TEAM1_MASK;
                bool visOwn = (cellVis & ownMask) != 0;
                // Also check own player bit specifically
                ushort playerBit = (ushort)(1 << i);
                bool visPlayer = (cellVis & playerBit) != 0;

                if (visOwn) pass++; else fail++;
                sb.AppendFormat("  H[{0}] w=({1:F0},{2:F0}) f=({3},{4}) reveal=0x{5:X4} blk=0x{6:X4} L2=0x{7:X4} L3=0x{8:X4} own={9} bit={10}\n",
                    i, hx, hy, fogX, fogY, cellVis, cellMask, cellL3, cellL4,
                    visOwn ? "Y" : "N", visPlayer ? "Y" : "N");
            }
            sb.AppendFormat("  Result: {0} pass, {1} fail", pass, fail);
            Log(sb.ToString());

            // If all fail, try alternative origin calculation
            if (fail > 0 && pass == 0)
            {
                Log("[RLComm] FOG: All validation failed! Trying alternative origins...");
                TryAlternativeFogOrigins();
            }
        }

        /// <summary>
        /// Try alternative fog origin calculations if the default one fails validation.
        /// </summary>
        private static void TryAlternativeFogOrigins()
        {
            if (!_fogInitialized) return;

            // Get first alive hero position for testing
            float testX = 0, testY = 0;
            int testPid = -1;
            for (int i = 0; i < MAX_PLAYERS; i++)
            {
                if (!heroRegistered[i]) continue;
                float hp = 0f;
                try { hp = Natives.GetUnitState(heroes[i], JassUnitState.Life); } catch { }
                if (hp <= 0.405f) continue;
                try { testX = Natives.GetUnitX(heroes[i]); } catch { }
                try { testY = Natives.GetUnitY(heroes[i]); } catch { }
                testPid = i;
                break;
            }
            if (testPid < 0) return;

            ushort playerBit = (ushort)(1 << testPid);

            // Try various common origin formulas
            float[][] origins = new float[][] {
                new float[] { -_fogGridW * _fogCellSize / 2f, -_fogGridH * _fogCellSize / 2f },  // centered at (0,0)
                new float[] { MAP_MIN_X, MAP_MIN_Y },  // camera bounds min
                new float[] { MAP_MIN_X - 512, MAP_MIN_Y - 512 },  // camera bounds - border
                new float[] { MAP_MIN_X - 256, MAP_MIN_Y - 256 },  // camera bounds - half border
            };

            for (int o = 0; o < origins.Length; o++)
            {
                float origX = origins[o][0], origY = origins[o][1];
                int fx = (int)((testX - origX) / _fogCellSize);
                int fy = (int)((testY - origY) / _fogCellSize);
                if (fx < 0 || fx >= _fogGridW || fy < 0 || fy >= _fogGridH)
                {
                    Log($"[RLComm] FOG ALT[{o}] origin=({origX},{origY}): OUT OF BOUNDS fog=({fx},{fy})");
                    continue;
                }
                int offset = ((fy << (_fogShift & 0x1F)) + fx) * 2;
                ushort cell = (ushort)Marshal.ReadInt16(_fogBitmapPtr + offset);
                bool match = (cell & playerBit) != 0;
                Log($"[RLComm] FOG ALT[{o}] origin=({origX},{origY}): fog=({fx},{fy}) cell=0x{cell:X4} match={match}");

                if (match)
                {
                    Log($"[RLComm] FOG: FOUND WORKING ORIGIN = ({origX}, {origY})!");
                    _fogOriginX = origX;
                    _fogOriginY = origY;
                    break;
                }
            }

            // Brute-force: scan nearby cells to find the hero
            Log($"[RLComm] FOG BRUTEFORCE: Looking for player bit 0x{playerBit:X4} near hero ({testX:F0},{testY:F0})");
            for (int searchRadius = 0; searchRadius <= 20; searchRadius++)
            {
                int baseX = (int)(testX / _fogCellSize);
                int baseY = (int)(testY / _fogCellSize);
                for (int dy = -searchRadius; dy <= searchRadius; dy++)
                {
                    for (int dx = -searchRadius; dx <= searchRadius; dx++)
                    {
                        if (Math.Abs(dx) != searchRadius && Math.Abs(dy) != searchRadius) continue;
                        int sx = baseX + dx, sy = baseY + dy;
                        if (sx < 0 || sx >= _fogGridW || sy < 0 || sy >= _fogGridH) continue;
                        int offset = ((sy << (_fogShift & 0x1F)) + sx) * 2;
                        ushort cell = (ushort)Marshal.ReadInt16(_fogBitmapPtr + offset);
                        if ((cell & playerBit) != 0)
                        {
                            float foundOriginX = testX - sx * _fogCellSize;
                            float foundOriginY = testY - sy * _fogCellSize;
                            Log($"[RLComm] FOG BRUTEFORCE: FOUND at fog=({sx},{sy}), implies origin=({foundOriginX:F1},{foundOriginY:F1})");
                            _fogOriginX = foundOriginX;
                            _fogOriginY = foundOriginY;
                            return;
                        }
                    }
                }
            }
            Log("[RLComm] FOG BRUTEFORCE: Not found within search radius!");
        }

        // ============================================================
        // Speed Table Patch
        // ============================================================
        private const int SPEED_TABLE_RVA = 0xCB355C;
        private static int[] _origSpeedTable = new int[3];
        private static bool _speedTablePatched;

        private static void ReadSpeedTable()
        {
            IntPtr gameDll = Kernel32.GetModuleHandleA("game.dll");
            if (gameDll == IntPtr.Zero) return;

            IntPtr tableAddr = gameDll + SPEED_TABLE_RVA;
            for (int i = 0; i < 3; i++)
            {
                _origSpeedTable[i] = Marshal.ReadInt32(tableAddr + i * 4);
            }

            Log($"[RLComm] SpeedTable at game.dll+0x{SPEED_TABLE_RVA:X} (0x{tableAddr.ToInt64():X8}):");
            Log($"[RLComm]   [0] = {_origSpeedTable[0]} (x 5ms = {_origSpeedTable[0] * 5}ms -- Slowest)");
            Log($"[RLComm]   [1] = {_origSpeedTable[1]} (x 5ms = {_origSpeedTable[1] * 5}ms -- Normal)");
            Log($"[RLComm]   [2] = {_origSpeedTable[2]} (x 5ms = {_origSpeedTable[2] * 5}ms -- Fastest)");
        }

        private static void PatchSpeedTable()
        {
            if (_speedMultiplier <= 1.0) return;

            IntPtr gameDll = Kernel32.GetModuleHandleA("game.dll");
            if (gameDll == IntPtr.Zero) return;

            IntPtr tableAddr = gameDll + SPEED_TABLE_RVA;

            ReadSpeedTable();

            int newFastest = Math.Max(1, (int)(_origSpeedTable[2] * _speedMultiplier));

            uint oldProt;
            if (!VirtualProtect(tableAddr, 12, PAGE_EXECUTE_READWRITE, out oldProt))
            {
                Log("[RLComm] SpeedTable: VirtualProtect failed!");
                return;
            }

            int newSlowest = Math.Max(1, (int)(_origSpeedTable[0] * _speedMultiplier));
            int newNormal = Math.Max(1, (int)(_origSpeedTable[1] * _speedMultiplier));

            Marshal.WriteInt32(tableAddr + 0, newSlowest);
            Marshal.WriteInt32(tableAddr + 4, newNormal);
            Marshal.WriteInt32(tableAddr + 8, newFastest);

            VirtualProtect(tableAddr, 12, oldProt, out _);

            _speedTablePatched = true;

            Log($"[RLComm] SpeedTable PATCHED ({_speedMultiplier}x):");
            Log($"[RLComm]   [0] {_origSpeedTable[0]} -> {newSlowest} ({newSlowest * 5}ms)");
            Log($"[RLComm]   [1] {_origSpeedTable[1]} -> {newNormal} ({newNormal * 5}ms)");
            Log($"[RLComm]   [2] {_origSpeedTable[2]} -> {newFastest} ({newFastest * 5}ms)");
        }

        /// <summary>Dump current values at all scan candidates (debug)</summary>
        private static void DumpScanState()
        {
            IntPtr gameDll = Kernel32.GetModuleHandleA("game.dll");
            Log($"[RLComm] SCANDUMP: {_scanCandidates.Count} candidates, {_turnIntervalAddrs.Count} turn-interval addrs");
            foreach (IntPtr addr in _scanCandidates)
            {
                try
                {
                    int val = Marshal.ReadInt32(addr);
                    long rva = addr.ToInt64() - gameDll.ToInt64();
                    string marker = _turnIntervalAddrs.Contains(addr) ? " ***" : "";
                    Log($"[RLComm]   game.dll+0x{rva:X} = {val}{marker}");
                }
                catch { }
            }
        }

        // ============================================================
        // Differential Memory Scan
        // ============================================================
        private static byte[] _diffSnapshot;
        private static long _diffBase;
        private static List<long> _diffRegionOffsets = new List<long>();
        private static List<int> _diffRegionLengths = new List<int>();

        private static byte[] _stormSnapshot;
        private static long _stormBase;
        private static List<long> _stormRegionOffsets = new List<long>();
        private static List<int> _stormRegionLengths = new List<int>();

        private static void SnapshotModule(string name, long imageSize,
            out IntPtr basePtr, out byte[] snapshot,
            List<long> regionOffsets, List<int> regionLengths)
        {
            basePtr = Kernel32.GetModuleHandleA(name);
            snapshot = null;
            regionOffsets.Clear();
            regionLengths.Clear();

            if (basePtr == IntPtr.Zero)
            {
                Log($"[RLComm] DIFFSCAN: {name} not found");
                return;
            }

            long baseAddr = basePtr.ToInt64();
            snapshot = new byte[imageSize];
            int mbiSize = Marshal.SizeOf(typeof(MEMORY_BASIC_INFORMATION));
            IntPtr scanAddr = basePtr;
            int totalBytes = 0;
            int regionCount = 0;

            while (scanAddr.ToInt64() < baseAddr + imageSize)
            {
                MEMORY_BASIC_INFORMATION mbi;
                if (VirtualQuery(scanAddr, out mbi, mbiSize) == 0) break;

                long regionSize = mbi.RegionSize.ToInt64();
                if (regionSize <= 0) break;

                long regionStart = mbi.BaseAddress.ToInt64();
                long regionEnd = regionStart + regionSize;

                long start = Math.Max(regionStart, baseAddr);
                long end = Math.Min(regionEnd, baseAddr + imageSize);

                if (end > start && mbi.State == MEM_COMMIT && (mbi.Protect & PAGE_GUARD) == 0)
                {
                    int offset = (int)(start - baseAddr);
                    int length = (int)(end - start);

                    try
                    {
                        Marshal.Copy(new IntPtr(start), snapshot, offset, length);
                        regionOffsets.Add(offset);
                        regionLengths.Add(length);
                        totalBytes += length;
                        regionCount++;
                    }
                    catch (Exception ex)
                    {
                        Log($"[RLComm] DIFFSCAN: {name} copy failed +0x{offset:X}: {ex.Message}");
                    }
                }

                scanAddr = new IntPtr(Math.Max(regionEnd, scanAddr.ToInt64() + 1));
            }

            Log($"[RLComm] DIFFSCAN PRE: {name}=0x{baseAddr:X8} {regionCount} regions {totalBytes} bytes");
        }

        private static void CompareSnapshot(string name, long baseAddr, byte[] snapshot,
            List<long> regionOffsets, List<int> regionLengths)
        {
            if (snapshot == null)
            {
                Log($"[RLComm] DIFFSCAN POST: {name} no snapshot");
                return;
            }

            int totalDiffs = 0;

            for (int ri = 0; ri < regionOffsets.Count; ri++)
            {
                long startOff = regionOffsets[ri];
                int length = regionLengths[ri];

                byte[] current = new byte[length];
                try
                {
                    Marshal.Copy(new IntPtr(baseAddr + startOff), current, 0, length);
                }
                catch { continue; }

                for (int i = 0; i + 4 <= length; i += 4)
                {
                    int oldVal = BitConverter.ToInt32(snapshot, (int)startOff + i);
                    int newVal = BitConverter.ToInt32(current, i);

                    if (oldVal != newVal)
                    {
                        totalDiffs++;
                        long rva = startOff + i;

                        string line = $"[RLComm] DIFF {name}+0x{rva:X}: {oldVal} (0x{(uint)oldVal:X8}) -> {newVal} (0x{(uint)newVal:X8})";

                        float fOld = BitConverter.ToSingle(snapshot, (int)startOff + i);
                        float fNew = BitConverter.ToSingle(current, i);
                        if (!float.IsNaN(fOld) && !float.IsInfinity(fOld) &&
                            !float.IsNaN(fNew) && !float.IsInfinity(fNew) &&
                            Math.Abs(fOld) < 1e15 && Math.Abs(fNew) < 1e15)
                        {
                            line += $" (float: {fOld:G6} -> {fNew:G6})";
                        }

                        ushort wOldLo = (ushort)(oldVal & 0xFFFF);
                        ushort wOldHi = (ushort)((oldVal >> 16) & 0xFFFF);
                        ushort wNewLo = (ushort)(newVal & 0xFFFF);
                        ushort wNewHi = (ushort)((newVal >> 16) & 0xFFFF);
                        if (wOldLo != wNewLo || wOldHi != wNewHi)
                        {
                            line += $" (w16: {wOldLo},{wOldHi} -> {wNewLo},{wNewHi})";
                        }

                        Log(line);

                        if (totalDiffs >= 500)
                        {
                            Log($"[RLComm] DIFF {name}: ... truncated (>500)");
                            goto done;
                        }
                    }
                }
            }

            done:
            Log($"[RLComm] DIFFSCAN POST: {name} {totalDiffs} DWORD changes");
        }

        private static void DiffScanPre()
        {
            IntPtr gameDllPtr;
            byte[] gameDllSnap;
            SnapshotModule("game.dll", 0xE00000, out gameDllPtr, out gameDllSnap,
                _diffRegionOffsets, _diffRegionLengths);
            _diffBase = gameDllPtr.ToInt64();
            _diffSnapshot = gameDllSnap;

            IntPtr stormPtr;
            byte[] stormSnap;
            SnapshotModule("Storm.dll", 0x100000, out stormPtr, out stormSnap,
                _stormRegionOffsets, _stormRegionLengths);
            _stormBase = stormPtr.ToInt64();
            _stormSnapshot = stormSnap;
        }

        private static void DiffScanPost()
        {
            CompareSnapshot("game.dll", _diffBase, _diffSnapshot,
                _diffRegionOffsets, _diffRegionLengths);
            CompareSnapshot("Storm.dll", _stormBase, _stormSnapshot,
                _stormRegionOffsets, _stormRegionLengths);

            _diffSnapshot = null;
            _stormSnapshot = null;
        }

        /// <summary>
        /// Override local player slot by patching CGameWar3.localPlayerId in Game.dll memory.
        /// This allows the human player to control any player slot (0-11) in single-player.
        /// </summary>
        private static void PatchLocalPlayerSlot(int desiredSlot)
        {
            if (desiredSlot < 0 || desiredSlot >= MAX_PLAYERS) return;
            try
            {
                IntPtr gameDll = Kernel32.GetModuleHandleA("game.dll");
                if (gameDll == IntPtr.Zero)
                {
                    Log("[RLComm] PatchLocalPlayer: Game.dll not found");
                    return;
                }

                // Read the CGameWar3 global pointer
                IntPtr ptrAddr = gameDll + GAME_WAR3_PTR_OFFSET;
                IntPtr gameWar3 = Marshal.ReadIntPtr(ptrAddr);
                if (gameWar3 == IntPtr.Zero)
                {
                    Log("[RLComm] PatchLocalPlayer: CGameWar3 ptr is null");
                    return;
                }

                // Read current localPlayerId
                short currentSlot = Marshal.ReadInt16(gameWar3 + LOCAL_PLAYER_ID_OFFSET);
                Log($"[RLComm] PatchLocalPlayer: current={currentSlot}, desired={desiredSlot}");

                if (currentSlot == (short)desiredSlot)
                {
                    Log("[RLComm] PatchLocalPlayer: already correct, skipping");
                    return;
                }

                // Write desired slot to localPlayerId (+0x28) and backup (+0x2a)
                Marshal.WriteInt16(gameWar3 + LOCAL_PLAYER_ID_OFFSET, (short)desiredSlot);
                Marshal.WriteInt16(gameWar3 + LOCAL_PLAYER_ID_BACKUP_OFFSET, (short)desiredSlot);

                // Verify
                short verify = Marshal.ReadInt16(gameWar3 + LOCAL_PLAYER_ID_OFFSET);
                Log($"[RLComm] PatchLocalPlayer: wrote slot {desiredSlot}, verify={verify}");
            }
            catch (Exception ex)
            {
                Log($"[RLComm] PatchLocalPlayer error: {ex.Message}");
            }
        }

        /// <summary>
        /// Patch Storm.dll SyncDelay to minimum (10ms) for maximum simulation speed.
        /// </summary>
        private static void ApplySyncDelay(int delayMs)
        {
            try
            {
                IntPtr stormDll = Kernel32.GetModuleHandleA("Storm.dll");
                if (stormDll == IntPtr.Zero)
                {
                    Log("[RLComm] Speed: Storm.dll not found");
                    return;
                }

                int value = delayMs < 10 ? 10 : delayMs > 550 ? 550 : delayMs;
                IntPtr ptr = Memory.FollowPointer(stormDll + 0x58330, 0x68DBD6C0);
                if (ptr == IntPtr.Zero)
                {
                    Log("[RLComm] Speed: SyncDelay pointer chain failed");
                    return;
                }

                ptr += 0x2F0;
                for (int i = 0; i <= 0x440; i += 0x220)
                    Memory.Patch(ptr + i, value);

                Log($"[RLComm] Speed: SyncDelay set to {value}ms");
            }
            catch (Exception ex)
            {
                Log($"[RLComm] Speed: SyncDelay error: {ex.Message}");
            }
        }

        // ============================================================
        // Hero Ability Table (Extended with D/F, maxCd, manaCost, orderId)
        // ============================================================

        private struct SkillInfo
        {
            public int abilId;       // FourCC ability id (used for mask: GetUnitAbilityLevel check)
            public string orderId;   // WC3 order string
            public int targetType;   // 0=immediate, 1=unit target, 2=point target
            public float[] maxCd;    // max cooldown per level (5 entries)
            public int[] manaCost;   // mana cost per level (5 entries)
            public float maxRange;   // point skill maximum range
            public int learnAbilId;  // if != 0, SelectHeroSkill uses this instead of abilId
                                     // e.g. FakeAssassin R: abilId=A01P (cast), learnAbilId=A04X (levelup)
            public int sharedCdGroup; // >0 means all skills with same group share cooldown
                                      // e.g. FakeAssassin QWER all have sharedCdGroup=1
        }

        private struct HeroData
        {
            public string heroId;      // type id string e.g. "H000"
            public float baseAtk;
            public float atkPerStr;
            public float baseDef;
            public float atkRange;
            public float baseAtkSpd;
            public int mainStat;       // 0=str, 1=agi, 2=int
            public SkillInfo[] skills; // Q=0, W=1, E=2, R=3, D=4, F=5
            public int[] attributeCost; // cost for attributes A,B,C,D
        }

        private static Dictionary<int, HeroData> _heroDataTable = new Dictionary<int, HeroData>();

        private static int FourCC(string s) { return (s[0] << 24) | (s[1] << 16) | (s[2] << 8) | s[3]; }

        private static SkillInfo MakeSkill(string abilIdStr, string orderId, int targetType,
            float[] maxCd = null, int[] manaCost = null, float maxRange = 600f,
            string learnAbilIdStr = null, int sharedCdGroup = 0)
        {
            return new SkillInfo
            {
                abilId = FourCC(abilIdStr),
                orderId = orderId,
                targetType = targetType,
                maxCd = maxCd ?? new float[] { 10, 9, 8, 7, 6 },
                manaCost = manaCost ?? new int[] { 100, 110, 120, 130, 140 },
                maxRange = maxRange,
                learnAbilId = learnAbilIdStr != null ? FourCC(learnAbilIdStr) : 0,
                sharedCdGroup = sharedCdGroup
            };
        }

        private static SkillInfo EmptySkill()
        {
            return new SkillInfo
            {
                abilId = 0,
                orderId = "",
                targetType = 0,
                maxCd = new float[] { 0, 0, 0, 0, 0 },
                manaCost = new int[] { 0, 0, 0, 0, 0 }
            };
        }

        private static void InitHeroDataTable()
        {
            _heroDataTable.Clear();

            // Per-hero attribute costs from JASS s__User_addAttribute calls
            // Format: { attr1, attr2, attr3, attr4 }

            // ---- Saber (H000) ----
            _heroDataTable[FourCC("H000")] = new HeroData
            {
                heroId = "H000", baseAtk = 30, atkPerStr = 2.0f, baseDef = 5, atkRange = 128, baseAtkSpd = 1.7f,
                mainStat = 0, attributeCost = new int[] { 7, 10, 14, 9 },
                skills = new SkillInfo[]
                {
                    MakeSkill("A087", "unavatar", 1),            // Q InvisibleAir (ANcl Ncl6=unavatar, unit)
                    MakeSkill("A01H", "unavengerform", 1),      // W Caliburn (ANcl Ncl6=unavengerform, unit)
                    MakeSkill("A01J", "carrionswarm", 2),       // E Excalibur (AUcs, point)
                    MakeSkill("A02B", "battleroar", 0),         // R Avalon (ANbr, imm/self-buff)
                    MakeSkill("A0G0", "tankloadpilot", 2),      // D StrikeAir (ANcl, lv1:point/tankloadpilot, lv2+:unit/unflamingarrows)
                    MakeSkill("A0A5", "berserk", 0),            // F SaberInstinct (Absk, imm; 2차속성 후 활성화)
                }
            };

            // ---- Archer/Emiya (H001) ----
            // Normal: QWER + D(OverEdge, 3rd attr) + F(Clairvoyance→Hrunting)
            // UBW: Q(간장막야) W(게이볼그) E(로아이아스유지) R(룰브레이커) F(보구투척) G(엑스칼리버이마쥬)
            // UBW+투영강화: Q W E R F(보구투척) B(전투영연속충사) + D(오버엣지 유지)
            _heroDataTable[FourCC("H001")] = new HeroData
            {
                heroId = "H001", baseAtk = 28, atkPerStr = 2.0f, baseDef = 4, atkRange = 600, baseAtkSpd = 1.8f,
                mainStat = 0, attributeCost = new int[] { 10, 13, 11, 13 },
                skills = new SkillInfo[]
                {
                    MakeSkill("A019", "transmute", 1),           // Q Kanshou&Bakuya (ANcl Ncl6=transmute, unit)
                    MakeSkill("A01B", "windwalk", 1),           // W BrokenPhantasm (ANcl Ncl6=windwalk, unit)
                    MakeSkill("A014", "blight", 0),             // E RhoAias (ANcl Ncl6=blight, imm/self-cast)
                    MakeSkill("A03C", "fanofknives", 0),        // R UBW (AEfk, imm)
                    MakeSkill("A00P", "inferno", 2),            // D OverEdge (AUin, point; 3rd attr + Q 5회 충전)
                    MakeSkill("A006", "farsight", 2),           // F Clairvoyance (AOfs, point; 땅 타겟→시야 확보)
                }
            };

            // ---- Lancer/Cu Chulainn (H002) ----
            // Main skills: A052(SwiftStrikes), A01K(GaeBolg), A028(FlyingSpear)
            // Runes (A01N learn → 5 added, shared 20s CD, only 1 active at a time):
            //   A035(Q재정비), A02T(W회복,NO ORDER), A03H(E함정), A05A(R해체,lv1 no Ncl6), A05B(A탐지)
            // Passive: A09P(화염의룬, ANic, auto on attack/death)
            // GaeBolg강화 속성: boolean flag only, same ability IDs
            _heroDataTable[FourCC("H002")] = new HeroData
            {
                heroId = "H002", baseAtk = 32, atkPerStr = 2.0f, baseDef = 5, atkRange = 128, baseAtkSpd = 1.6f,
                mainStat = 0, attributeCost = new int[] { 7, 13, 9, 9 },
                skills = new SkillInfo[]
                {
                    MakeSkill("A01N", "", 0),                   // Q RuneMaster (hero skill, learn-only, 캐스팅 불가)
                    MakeSkill("A052", "berserk", 0),            // W SwiftStrikes (Absk, imm/self-buff)
                    MakeSkill("A01K", "web", 1),                // E GaeBolg (ANcl Ncl6=web, unit)
                    MakeSkill("A028", "clusterrockets", 2),     // R FlyingSpear (ANcs, point)
                    MakeSkill("A035", "carrionscarabs", 0),     // D Rune-재정비 (ANcl Ncl6=carrionscarabs, imm)
                    MakeSkill("A03H", "stasistrap", 2),         // F Rune-함정 (Asta, point)
                    // Rune-탐지(A05B,"charm",0) 7th slot 부족으로 제외
                    // Excluded: A02T(회복,NO ORDER), A05A(해체,lv1 no Ncl6)
                }
            };

            // ---- Rider/Medusa (H003) ----
            // In-game: Q=단검폭사(A01Q), W=암흑신전(A01R), E=선혈신전(A01F), R=기영의고삐(A01E)
            _heroDataTable[FourCC("H003")] = new HeroData
            {
                heroId = "H003", baseAtk = 26, atkPerStr = 2.0f, baseDef = 4, atkRange = 128, baseAtkSpd = 1.7f,
                mainStat = 1, attributeCost = new int[] { 8, 15, 11, 11 },
                skills = new SkillInfo[]
                {
                    MakeSkill("A01Q", "fanofknives", 0),        // Q 단검폭사 CatenaSword (AEfk, imm)
                    MakeSkill("A01R", "carrionswarm", 2),       // W 암흑신전 BreakerGorgon (AUcs, point)
                    MakeSkill("A01F", "battleroar", 0),         // E 선혈신전 BloodFort (ANbr, imm)
                    MakeSkill("A01E", "inferno", 2),            // R 기영의고삐 Bellerophon (AUin, point)
                    EmptySkill(),                                // D (none)
                    EmptySkill(),                                // F (none)
                }
            };

            // ---- Caster/Medea (H004) ----
            // In-game: Q=마력방패, W=신대의마술(스펠북), E=룰브레이커, R=헤카틱글레이어, D=진지작성, F=도구작성
            // SelectHeroSkill: A022, A08D, A08A, A06L
            // W(A022)=Aspb 스펠북(캐스팅불가), D(A049)=AIbt 빌드, F(Agyv)=아이템판매UI → RL 직접사용 불가
            // 스펠북 하위스킬: A03L(화염장막), A01I(무력화), A00L(고속신언), A08E(새크리파이스), A00Y(마력전달)
            _heroDataTable[FourCC("H004")] = new HeroData
            {
                heroId = "H004", baseAtk = 25, atkPerStr = 2.0f, baseDef = 3, atkRange = 600, baseAtkSpd = 1.8f,
                mainStat = 2, attributeCost = new int[] { 12, 9, 6, 15 },
                skills = new SkillInfo[]
                {
                    MakeSkill("A022", "", 0),                    // Q 신대의마술 (Aspb 스펠북, learn-only)
                    MakeSkill("A08D", "channel", 0),            // W 마력방패 Aegis (ANcl, imm)
                    MakeSkill("A08A", "unbearform", 1),         // E 룰브레이커 RuleBreaker (ANcl, unit)
                    MakeSkill("A06L", "healingspray", 2),       // R 헤카틱글레이어 HecaticGraea (ANhs, point)
                    MakeSkill("A03L", "roar", 0),               // D 화염장막 FireWall (ANcl Ncl6=roar, imm) 스펠북sub
                    MakeSkill("A01I", "silence", 2),            // F 무력화 Silence (ACsi, point) 스펠북sub
                    // 슬롯부족 제외: A00L(고속신언,blizzard,point), A08E(새크리파이스,phoenixmorph,imm), A00Y(마력전달,slow,unit)
                    // 비RL: A049(진지작성,AIbt빌드), Agyv(도구작성,판매UI)
                }
            };

            // ---- FakeAssassin (H005) ----
            // In-game: Q=문지기, W=소화의소양, E=섬풍, R=츠바메가에시, D=발도(4차속성)
            _heroDataTable[FourCC("H005")] = new HeroData
            {
                heroId = "H005", baseAtk = 30, atkPerStr = 2.0f, baseDef = 3, atkRange = 128, baseAtkSpd = 1.5f,
                mainStat = 1, attributeCost = new int[] { 6, 16, 11, 12 },
                skills = new SkillInfo[]
                {
                    MakeSkill("A01L", "chemicalrage", 0, manaCost: new int[]{0,0,0,0,0}, sharedCdGroup: 1),  // Q 문지기 Gatekeeper (ANcr, imm/self-buff) shared CD
                    MakeSkill("A05D", "battleroar", 0, manaCost: new int[]{0,0,0,0,0}, sharedCdGroup: 1),   // W 소화의소양 Knowledge (ANbr, imm/self-buff) shared CD
                    MakeSkill("A01O", "thunderclap", 0, manaCost: new int[]{0,0,0,0,0}, sharedCdGroup: 1),  // E 섬풍 Windblade (AHtc, imm/self-AOE) shared CD
                    MakeSkill("A01P", "fingerofdeath", 1, manaCost: new int[]{0,0,0,0,0}, learnAbilIdStr: "A04X", sharedCdGroup: 1), // R 츠바메가에시 TsubameGaeshi (ANfd, unit; learn=A04X proxy) shared CD
                    MakeSkill("A011", "bloodlustoff", 2, manaCost: new int[]{0,0,0,0,0}),       // D 발도 QuickDraw (ANcl Ncl6=bloodlustoff, point; 4차속성)
                    EmptySkill(),                                // F (none)
                }
            };

            // ---- Berserker (H006) ----
            // In-game: Q=참격, W=표호, E=광화, R=나인라이브즈. 속성 전부 패시브.
            _heroDataTable[FourCC("H006")] = new HeroData
            {
                heroId = "H006", baseAtk = 35, atkPerStr = 2.0f, baseDef = 6, atkRange = 128, baseAtkSpd = 1.9f,
                mainStat = 0, attributeCost = new int[] { 16, 13, 17, 12 },
                skills = new SkillInfo[]
                {
                    MakeSkill("A04J", "carrionswarm", 2),        // Q 참격 TrueStrike (AUcs, point)
                    MakeSkill("A00Z", "berserk", 0),            // W 표호 Bravery (Absk, imm/self-buff)
                    MakeSkill("A01D", "battleroar", 0),         // E 광화 MadEnhancement (ANbr, imm/self-buff)
                    MakeSkill("A015", "clusterrockets", 2),     // R 나인라이브즈 NineLives (ANcs, point)
                    EmptySkill(),                                // D (none)
                    EmptySkill(),                                // F (none)
                }
            };

            // ---- SaberAlter (H007) ----
            // In-game: Q=폭정, W=마력방출, E=비왕철퇴, R=ExcaliburMorgan, F=흉폭성개방
            // 속성: 데미지/범위 강화만, 액티브 변경 없음
            _heroDataTable[FourCC("H007")] = new HeroData
            {
                heroId = "H007", baseAtk = 33, atkPerStr = 2.0f, baseDef = 5, atkRange = 128, baseAtkSpd = 1.8f,
                mainStat = 0, attributeCost = new int[] { 13, 15, 12, 10 },
                skills = new SkillInfo[]
                {
                    MakeSkill("A00B", "roar", 0),                // Q 폭정 Tyrant (Aroa, imm/self-buff)
                    MakeSkill("A024", "thunderclap", 0),        // W 마력방출 PranaBurst (AHtc, imm/self-AOE)
                    MakeSkill("A07S", "recharge", 2),           // E 비왕철퇴 Vortigern (ANcl Ncl6=recharge, point)
                    MakeSkill("A023", "carrionswarm", 2),       // R ExcaliburMorgan (AUcs, point)
                    EmptySkill(),                                // D (none)
                    MakeSkill("A01T", "channel", 0),            // F 흉폭성개방 Ferocity (ANcl Ncl6=channel, imm)
                }
            };

            // ---- TrueAssassin (H008) ----
            // In-game: Q=기습, W=자기개조, E=강탈, R=망상심음, D=단도투척, F=기척차단
            _heroDataTable[FourCC("H008")] = new HeroData
            {
                heroId = "H008", baseAtk = 28, atkPerStr = 2.0f, baseDef = 3, atkRange = 128, baseAtkSpd = 1.5f,
                mainStat = 1, attributeCost = new int[] { 9, 8, 9, 13 },
                skills = new SkillInfo[]
                {
                    MakeSkill("A012", "windwalk", 0),           // Q 기습 Ambush (AOwk, imm/self-buff)
                    MakeSkill("A018", "starfall", 0),           // W 자기개조 SelfReconstruction (AEsb, imm)
                    MakeSkill("A09Y", "fingerofdeath", 1),      // E 강탈 Steal (ANfd, unit)
                    MakeSkill("A02A", "unavengerform", 1),      // R 망상심음 Zabaniya (ANcl, unit)
                    MakeSkill("A03A", "forkedlightning", 1),    // D 단도투척 ThrowDagger (ANfl, unit)
                    MakeSkill("A009", "shadowmeld", 0),         // F 기척차단 PresenceConcealment (Ashm, imm/toggle)
                }
            };

            // ---- Gilgamesh (H009) ----
            // In-game: Q=하늘의사슬, W=마르두크, E=왕의재보, R=에아, F=보구투척
            _heroDataTable[FourCC("H009")] = new HeroData
            {
                heroId = "H009", baseAtk = 30, atkPerStr = 2.0f, baseDef = 4, atkRange = 500, baseAtkSpd = 1.7f,
                mainStat = 0, attributeCost = new int[] { 8, 12, 10, 9 },
                skills = new SkillInfo[]
                {
                    MakeSkill("A01X", "magicleash", 1),         // Q 하늘의사슬 Enkidu (Amls, unit)
                    MakeSkill("A01Y", "unavengerform", 1),      // W 마르두크 Marduk (ANcl, unit)
                    MakeSkill("A01Z", "stampede", 2),           // E 왕의재보 GateOfBabylon (ANst, point)
                    MakeSkill("A02E", "carrionswarm", 2),       // R 에아 EnumaElish (AUcs, point)
                    EmptySkill(),                                // D (none)
                    MakeSkill("A01G", "blizzard", 2),           // F 보구투척 NPBarrage (ACbz, point)
                }
            };

            // ---- Avenger (H028) ----
            // In-game: Q=사멸원망, W=타루와이제레치에, E=잔해화, R=VergAvesta, F=어둠의순례
            // 잔해화 모드(morph): Q→A06U(무한의잔해,animatedead,imm), W→A080(피의저주,curse,unit)
            // uhab stubs: A0BG/A0BF/A0BH/A0BE (learn-only). Cast IDs below are correct.
            _heroDataTable[FourCC("H028")] = new HeroData
            {
                heroId = "H028", baseAtk = 29, atkPerStr = 2.0f, baseDef = 4, atkRange = 128, baseAtkSpd = 1.7f,
                mainStat = 0, attributeCost = new int[] { 8, 9, 15, 10 },
                skills = new SkillInfo[]
                {
                    MakeSkill("A06J", "roar", 0),                // Q 사멸원망 KillingIntent (ACro, imm)
                    MakeSkill("A05V", "polymorph", 1),          // W 타루와이제레치에 TawrichZarich (ANcl, unit)
                    MakeSkill("A05X", "chemicalrage", 0),       // E 잔해화 Shade (ANcr, imm/morph toggle)
                    MakeSkill("A02P", "absorb", 0),             // R VergAvesta (ANcl, imm)
                    MakeSkill("A0D7", "spellshield", 0),        // D 분진폭발 DustExplosion (ANcl, imm) - 잔해화 전용
                    MakeSkill("A045", "blink", 2),              // F 어둠의순례 DarkPilgrimage (AIbk, point)
                }
            };

            // ---- Lancelot (H03M) ----
            // In-game: Q=서브머신건, W=양날의검, E=기사는맨손으로(KoH스펠북), R=무훼의호광(Arondight)
            // KoH 하위스킬: A05M(Caliburn), A08W(Rhongomyniad), A090(RuleBreaker), A0SY(GaeDearg), A08R(ArondightOverload)
            // SelectHeroSkill: A02Z, A08F, A09B, A08S
            _heroDataTable[FourCC("H03M")] = new HeroData
            {
                heroId = "H03M", baseAtk = 32, atkPerStr = 2.0f, baseDef = 5, atkRange = 128, baseAtkSpd = 1.7f,
                mainStat = 0, attributeCost = new int[] { 11, 7, 7, 7 },
                skills = new SkillInfo[]
                {
                    MakeSkill("A02Z", "charm", 2),               // Q 서브머신건 SubmachineGun (ANcl, point)
                    MakeSkill("A08F", "berserk", 0),            // W 양날의검 DoubleEdgedSword (Absk, imm)
                    MakeSkill("A09B", "", 0),                   // E 기사는맨손으로 KnightOfHonor (learn-only, KoH스펠북)
                    MakeSkill("A08S", "chemicalrage", 0),       // R 무훼의호광 Arondight (ANcr, imm)
                    MakeSkill("A091", "lightningshield", 2),    // D KnightOfHonor2 (ANcl, point) 콤보 전용
                    MakeSkill("A0E6", "unroot", 0),             // F 무궁의무련 EternalArmsMastership (ANcl, imm) 1차속성
                    // KoH 하위(슬롯부족): A05M(Caliburn,unavengerform,unit), A08W(Rhongo,unflamingarrows,point)
                    // A090(RuleBreaker,unbearform,unit), A0SY(GaeDearg,autoharvestgold,unit), A08R(ArondightOverload,clusterrockets,point)
                }
            };

            // ---- Diarmuid (H04D) ----
            // In-game: Q=격돌, W=기사의무략, E=필멸의황장미(GaeBuidhe), R=파마의홍장미(GaeDearg), D=매혹의점, F=쌍창격(토글)
            // SelectHeroSkill: A0AG, A0AI, A0AO, A0AP
            // uhab stubs: A0AO/A0AP (learn-only). Cast IDs A0AL/A0AM are correct.
            // F(A0AX)=ACim(Immolation) 토글, 쌍창격 속성 찍을시 해금. GaeBuidhe 시전시 A0AX>0이면 자동 발동.
            _heroDataTable[FourCC("H04D")] = new HeroData
            {
                heroId = "H04D", baseAtk = 30, atkPerStr = 2.0f, baseDef = 4, atkRange = 128, baseAtkSpd = 1.6f,
                mainStat = 1, attributeCost = new int[] { 7, 11, 11, 14 },
                skills = new SkillInfo[]
                {
                    MakeSkill("A0AG", "chainlightning", 1),      // Q 격돌 Crash (AOcl, unit)
                    MakeSkill("A0AI", "stomp", 0),              // W 기사의무략 DoubleSpearMastery (AOw2, imm)
                    MakeSkill("A0AL", "frostnova", 1),          // E 필멸의황장미 GaeBuidhe (AUfn, unit)
                    MakeSkill("A0AM", "fingerofdeath", 1),      // R 파마의홍장미 GaeDearg (Afod, unit)
                    MakeSkill("A0B0", "drunkenhaze", 1),        // D 매혹의점 LoveSpotOfSeduction (Acdh, unit)
                    MakeSkill("A0AX", "immolation", 0),         // F 쌍창격 DoubleSpearStrike (ACim, toggle) 4차속성
                }
            };

            Log($"[RLComm] HeroDataTable: {_heroDataTable.Count} heroes loaded");
        }

        // ============================================================
        // Shop Items Table
        // ============================================================
        // item_buy indices: 0=none, 1-6=faire (기사회생) options, 7-17=gold shop items
        // Total: 18 indices → uint32 mask
        private struct ShopItem
        {
            public string typeId;
            public int cost;
        }

        private static readonly ShopItem[] _shopItems = new ShopItem[]
        {
            new ShopItem { typeId = "",     cost = 0 },     // 0:  none
            new ShopItem { typeId = "",     cost = 0 },     // 1:  faire - A04B 골드토큰(I00D)     order=flare
            new ShopItem { typeId = "",     cost = 0 },     // 2:  faire - A04F +3레벨             order=heal
            new ShopItem { typeId = "",     cost = 0 },     // 3:  faire - A0DD +12스탯포인트       order=sanctuary
            new ShopItem { typeId = "",     cost = 0 },     // 4:  faire - A04C 항마력(I00W)       order=frenzy
            new ShopItem { typeId = "",     cost = 0 },     // 5:  faire - A04D 아이템(I004)+3000g order=frostnova
            new ShopItem { typeId = "",     cost = 0 },     // 6:  faire - A00U 무적결계(I006)     order=holybolt
            new ShopItem { typeId = "I00E", cost = 100  },  // 7:  평행이동 주문서
            new ShopItem { typeId = "I00A", cost = 250  },  // 8:  속도의 보석
            new ShopItem { typeId = "I002", cost = 300  },  // 9:  패밀리어
            new ShopItem { typeId = "I01A", cost = 400  },  // 10: 조합 주문서
            new ShopItem { typeId = "I00B", cost = 500  },  // 11: 워프 포탈
            new ShopItem { typeId = "I00I", cost = 700  },  // 12: 회복포션
            new ShopItem { typeId = "I00G", cost = 750  },  // 13: 버서크 포션
            new ShopItem { typeId = "I003", cost = 800  },  // 14: 결계 주문서
            new ShopItem { typeId = "I011", cost = 1500 },  // 15: 네비게이션
            new ShopItem { typeId = "I00M", cost = 1500 },  // 16: 단체 워프 포탈
            new ShopItem { typeId = "I000", cost = 150  },  // 17: C랭크 마술주문서
        };

        // ============================================================
        // Seal Types
        // ============================================================
        // seal_use indices: 0=none, 1=first_seal_activate, 2=cd_reset, 3=hp_recover, 4=mp_recover, 5=revive, 6=teleport
        private static readonly int[] _sealCosts = new int[] { 0, 2, 2, 2, 2, 2, 2 };
        private static readonly int[] _sealCostsFirstActive = new int[] { 0, 2, 1, 1, 1, 1, 1 };

        // ============================================================
        // Buff Detection -- known ability IDs for common buffs
        // ============================================================
        private static readonly int _buffStun = FourCC("BPSE");     // Stunned
        private static readonly int _buffSlow = FourCC("Bslo");     // Slowed
        private static readonly int _buffSilence = FourCC("BNsi");  // Silence
        private static readonly int _buffRoot = FourCC("Bena");     // Entangle
        private static readonly int _buffInvuln = FourCC("Bvul");   // Invulnerable

        // ============================================================
        // Utility Functions
        // ============================================================

        /// <summary>
        /// Convert WC3 4-char type ID (int) to string.
        /// E.g. 0x48303030 -> "H000"
        /// </summary>
        private static Dictionary<int, string> _typeIdStringCache = new Dictionary<int, string>();
        private static Dictionary<int, byte[]> _typeIdBytesCache = new Dictionary<int, byte[]>();
        private static string TypeIdToString(int typeId)
        {
            string cached;
            if (_typeIdStringCache.TryGetValue(typeId, out cached))
                return cached;
            if (typeId == 0) { _typeIdStringCache[0] = "0000"; return "0000"; }
            char[] chars = new char[4];
            chars[0] = (char)((typeId >> 24) & 0xFF);
            chars[1] = (char)((typeId >> 16) & 0xFF);
            chars[2] = (char)((typeId >> 8) & 0xFF);
            chars[3] = (char)(typeId & 0xFF);
            string result = new string(chars);
            _typeIdStringCache[typeId] = result;
            _typeIdBytesCache[typeId] = Encoding.ASCII.GetBytes(result);
            return result;
        }
        private static byte[] TypeIdToBytes(int typeId)
        {
            byte[] cached;
            if (_typeIdBytesCache.TryGetValue(typeId, out cached))
                return cached;
            TypeIdToString(typeId); // populate cache
            return _typeIdBytesCache[typeId];
        }

        /// <summary>Clamp a float value between min and max (no Math.Clamp in .NET 4.6.2)</summary>
        private static float Clampf(float val, float min, float max)
        {
            return Math.Max(min, Math.Min(max, val));
        }

        private static int Clampi(int val, int min, int max)
        {
            return Math.Max(min, Math.Min(max, val));
        }

        /// <summary>Decode polar-coordinate action outputs into world-space point.
        /// pointX in [-1,1] maps to angle [-pi, pi], pointY in [-1,1] maps to distance [0, maxRange].</summary>
        private static void DecodePolarPoint(float pointX, float pointY, float maxRange,
            float cx, float cy, out float tx, out float ty)
        {
            float angle = pointX * (float)Math.PI;                // [-1,1] -> [-pi, pi]
            float dist  = (pointY + 1f) * 0.5f * maxRange;       // [-1,1] -> [0, maxRange]
            tx = cx + dist * (float)Math.Cos(angle);
            ty = cy + dist * (float)Math.Sin(angle);
            tx = Clampf(tx, MAP_MIN_X + 64, MAP_MAX_X - 64);
            ty = Clampf(ty, MAP_MIN_Y + 64, MAP_MAX_Y - 64);
        }

        /// <summary>Get player gold via PlayerState</summary>
        private static int GetPlayerGold(int pid)
        {
            try
            {
                JassPlayer p = Natives.Player(pid);
                return (int)Natives.GetPlayerState(p, Natives.ConvertPlayerState(1));
            }
            catch { return 0; }
        }

        /// <summary>Set player gold via PlayerState</summary>
        private static void SetPlayerGold(int pid, int amount)
        {
            try
            {
                JassPlayer p = Natives.Player(pid);
                Natives.SetPlayerState(p, Natives.ConvertPlayerState(1), amount);
            }
            catch { }
        }

        /// <summary>Check if a unit is alive (HP > 0.405)</summary>
        private static bool IsUnitAlive(JassUnit u)
        {
            try
            {
                float hp = Natives.GetUnitState(u, JassUnitState.Life);
                return hp > 0.405f;
            }
            catch { return false; }
        }

        /// <summary>Check if unit has an empty inventory slot</summary>
        private static bool HasEmptyItemSlot(JassUnit u)
        {
            try
            {
                for (int slot = 0; slot < 6; slot++)
                {
                    JassItem itm = Natives.UnitItemInSlot(u, slot);
                    if (itm.Handle == IntPtr.Zero) return true;
                    int itmType = (int)Natives.GetItemTypeId(itm);
                    if (itmType == 0) return true;
                }
            }
            catch { }
            return false;
        }

        /// <summary>Get seal charges from grail unit slot 1 (JASS: s__User_grail[pid] slot 1)</summary>
        private static SealResult GetSealItem(int pid)
        {
            if (!_grailRegistered[pid])
                return new SealResult { item = default(JassItem), charges = 0 };
            try
            {
                JassItem itm = Natives.UnitItemInSlot(_grailUnits[pid], 1); // slot 1 = command seal charges
                if (itm.Handle != IntPtr.Zero)
                {
                    int charges = (int)Natives.GetItemCharges(itm);
                    return new SealResult { item = itm, charges = charges };
                }
            }
            catch { }
            return new SealResult { item = default(JassItem), charges = 0 };
        }

        /// <summary>Get stat_points from grail unit slot 0 charges</summary>
        private static int _statPointsLogCounter = 0;
        private static bool _statPointsFirstLog = true;
        private static int GetStatPoints(int pid)
        {
            if (!_grailRegistered[pid])
            {
                if (pid == 0 && _statPointsFirstLog)
                {
                    _statPointsFirstLog = false;
                    Log($"[RLComm] DEBUG GetStatPoints p0: grailRegistered=FALSE");
                }
                return 0;
            }
            try
            {
                JassItem itm = Natives.UnitItemInSlot(_grailUnits[pid], 0);
                if (itm.Handle != IntPtr.Zero)
                {
                    int charges = (int)Natives.GetItemCharges(itm);
                    if (pid == 0 && (_statPointsFirstLog || ++_statPointsLogCounter % 500 == 0))
                    {
                        _statPointsFirstLog = false;
                        Log($"[RLComm] DEBUG GetStatPoints p{pid}: grail={_grailUnits[pid].Handle}, item={itm.Handle}, charges={charges}");
                    }
                    return charges;
                }
                else
                {
                    if (pid == 0 && (_statPointsFirstLog || ++_statPointsLogCounter % 500 == 0))
                    {
                        _statPointsFirstLog = false;
                        Log($"[RLComm] DEBUG GetStatPoints p{pid}: grail={_grailUnits[pid].Handle}, slot0 item=NULL");
                    }
                }
            }
            catch (Exception ex)
            {
                if (pid == 0)
                    Log($"[RLComm] DEBUG GetStatPoints p{pid} exception: {ex.Message}");
            }
            return 0;
        }

        /// <summary>Decrement stat_points (grail slot 0 charges) by 1</summary>
        private static bool DecrStatPoints(int pid)
        {
            if (!_grailRegistered[pid]) return false;
            try
            {
                JassItem itm = Natives.UnitItemInSlot(_grailUnits[pid], 0);
                if (itm.Handle != IntPtr.Zero)
                {
                    int charges = (int)Natives.GetItemCharges(itm);
                    if (charges > 0)
                    {
                        Natives.SetItemCharges(itm, charges - 1);
                        return true;
                    }
                }
            }
            catch { }
            return false;
        }

        /// <summary>Map team-relative ally index to absolute player index</summary>
        private static int ResolveAllyIndex(int heroIdx, int relativeIdx)
        {
            // relativeIdx 1-5 maps to teammates (excluding self)
            int teamBase = heroIdx < 6 ? 0 : 6;
            int count = 0;
            for (int p = teamBase; p < teamBase + 6; p++)
            {
                if (p == heroIdx) continue;
                count++;
                if (count == relativeIdx) return p;
            }
            return -1;
        }

        // Spawn positions for revive (approximate team spawn areas)
        private static readonly float[] _spawnX = { -7000f, -7000f, -7000f, -7000f, -7000f, -7000f,
                                                      7000f,  7000f,  7000f,  7000f,  7000f,  7000f };
        private static readonly float[] _spawnY = {  2000f,  2000f,  2000f,  2000f,  2000f,  2000f,
                                                      2000f,  2000f,  2000f,  2000f,  2000f,  2000f };

        // ============================================================
        // Preloader Override -- JASS command interface
        // ============================================================
        private delegate void PreloaderDelegate(JassStringArg filename);

        private static int _preloaderCallCount;
        private static void PreloaderOverride(JassStringArg filename)
        {
            string cmd = (string)filename;
            _preloaderCallCount++;
            // DEBUG: log EVERY Preloader call to find why RL_KILL never arrives
            if (_preloaderCallCount <= 200 || (cmd != null && cmd.StartsWith("RL")))
                Log($"[RLComm] Preloader#{_preloaderCallCount}: \"{cmd}\"");
            if (cmd == null || cmd.Length == 0) return;
            if (!cmd.StartsWith("RL")) return;

            try
            {
                // ---- New handlers (JASS state cache) ----
                if (cmd.StartsWith("RL_PDAT|"))
                {
                    // Format: "RL_PDAT|pid|u0,u1,...,u8|sealCd|firstActive|firstCdDown|attrCount"
                    string[] parts = cmd.Split('|');
                    if (parts.Length >= 7)
                    {
                        int pid = int.Parse(parts[1]);
                        if (pid >= 0 && pid < MAX_PLAYERS)
                        {
                            string[] upgs = parts[2].Split(',');
                            for (int u = 0; u < 9 && u < upgs.Length; u++)
                                JassStateCache.upgrades[pid, u] = int.Parse(upgs[u]);
                            JassStateCache.sealCd[pid] = int.Parse(parts[3]);
                            JassStateCache.firstActive[pid] = int.Parse(parts[4]);
                            JassStateCache.firstRemain[pid] = int.Parse(parts[5]);
                            JassStateCache.attrCount[pid] = int.Parse(parts[6]);
                        }
                    }
                }
                else if (cmd.StartsWith("RL_GREG|"))
                {
                    // Format: "RL_GREG|pid|handleId"
                    string[] parts = cmd.Split('|');
                    if (parts.Length >= 3)
                    {
                        int pid = int.Parse(parts[1]);
                        int hid = int.Parse(parts[2]);
                        if (pid >= 0 && pid < MAX_PLAYERS)
                        {
                            _grailUnits[pid] = new JassUnit(new IntPtr(hid));
                            _grailRegistered[pid] = true;
                            Log($"[RLComm] Grail registered: p{pid} hid={hid}");
                        }
                    }
                }
                else if (cmd.StartsWith("RL_SCORE|"))
                {
                    // Format: "RL_SCORE|score0|score1"
                    string[] parts = cmd.Split('|');
                    if (parts.Length >= 3)
                    {
                        JassStateCache.teamScore[0] = int.Parse(parts[1]);
                        JassStateCache.teamScore[1] = int.Parse(parts[2]);
                    }
                }
                else if (cmd.StartsWith("RL_KILL|"))
                {
                    // Format: "RL_KILL|killerPid|victimPid"
                    string[] parts = cmd.Split('|');
                    if (parts.Length >= 3)
                    {
                        int killer = int.Parse(parts[1]);
                        int victim = int.Parse(parts[2]);
                        _eventQueue.Add(new JObject
                        {
                            ["type"] = "KILL",
                            ["killer"] = killer,
                            ["victim"] = victim,
                            ["tick"] = tickCount
                        });

                        // Track PVP deaths for faire: only count if killer != victim (not suicide)
                        // and both are valid players (0-11, not creep=12+)
                        if (killer >= 0 && killer < MAX_PLAYERS && victim >= 0 && victim < MAX_PLAYERS && killer != victim)
                        {
                            int killerTeam = killer < 6 ? 0 : 1;
                            int victimTeam = victim < 6 ? 0 : 1;
                            if (killerTeam != victimTeam)  // enemy kill only
                            {
                                _advCount[victim]++;
                                if (_advCount[victim] % 7 == 0)
                                {
                                    Log($"[RLComm] Faire available for p{victim} (advCount={_advCount[victim]})");
                                }
                            }
                        }
                    }
                }
                else if (cmd.StartsWith("RL_CREEP|"))
                {
                    // Format: "RL_CREEP|killerPid"
                    string[] parts = cmd.Split('|');
                    if (parts.Length >= 2)
                    {
                        int killer = int.Parse(parts[1]);
                        _eventQueue.Add(new JObject
                        {
                            ["type"] = "CREEP_KILL",
                            ["killer"] = killer,
                            ["tick"] = tickCount
                        });
                    }
                }
                else if (cmd.StartsWith("RL_LVUP|"))
                {
                    // Format: "RL_LVUP|pid|newLevel"
                    string[] parts = cmd.Split('|');
                    if (parts.Length >= 3)
                    {
                        int pid = int.Parse(parts[1]);
                        int newLevel = int.Parse(parts[2]);
                        _eventQueue.Add(new JObject
                        {
                            ["type"] = "LEVEL_UP",
                            ["unit_idx"] = pid,
                            ["new_level"] = newLevel,
                            ["tick"] = tickCount
                        });
                    }
                }
                else if (cmd.StartsWith("RL_PORTAL|"))
                {
                    // Format: "RL_PORTAL|pid"
                    string[] parts = cmd.Split('|');
                    if (parts.Length >= 2)
                    {
                        int pid = int.Parse(parts[1]);
                        _eventQueue.Add(new JObject
                        {
                            ["type"] = "PORTAL",
                            ["unit_idx"] = pid,
                            ["tick"] = tickCount
                        });
                    }
                }
                else if (cmd.StartsWith("RL_ALARM|"))
                {
                    // Format: "RL_ALARM|pid"
                    string[] parts = cmd.Split('|');
                    if (parts.Length >= 2)
                    {
                        int pid = int.Parse(parts[1]);
                        if (pid >= 0 && pid < MAX_PLAYERS)
                            _alarmState[pid] = true;
                    }
                }
                // ---- Hero registration ----
                else if (cmd.StartsWith("RL_HREG|"))
                {
                    // Format: "RL_HREG|playerID|handleID"
                    string[] parts = cmd.Split('|');
                    if (parts.Length >= 3)
                    {
                        int pid = int.Parse(parts[1]);
                        int hid = int.Parse(parts[2]);
                        if (pid >= 0 && pid < MAX_PLAYERS && !heroRegistered[pid])
                        {
                            heroes[pid] = new JassUnit(new IntPtr(hid));
                            heroHandleIds[pid] = hid;
                            try { heroTypeIds[pid] = (int)Natives.GetUnitTypeId(new JassUnit(new IntPtr(hid))); } catch { }
                            heroRegistered[pid] = true;
                            heroCount++;
                            Log($"[RLComm] Hero registered: p{pid} hid={hid} type={TypeIdToString(heroTypeIds[pid])} ({heroCount}/{MAX_PLAYERS})");
                            if (heroCount >= MAX_PLAYERS)
                            {
                                heroesReady = true;
                                Log("[RLComm] All 12 heroes registered!");
                            }
                        }
                    }
                }
                else if (cmd.StartsWith("RL_DBG|"))
                {
                    Log($"[RLComm] DBG: {cmd}");
                }
                else if (cmd == "RL_TICK")
                {
                    OnTick();
                }
                else if (cmd == "RL_PING")
                {
                    Log("[RLComm] PING received - Preloader override confirmed working!");
                }
                else if (cmd.StartsWith("RL_DONE|"))
                {
                    // Format: "RL_DONE|winTeam" - JASS detected score=70
                    string[] parts = cmd.Split('|');
                    if (parts.Length >= 2 && episodeState == 0)
                    {
                        episodeState = 4; // score reached
                        Log($"[RLComm] RL_DONE: winTeam={parts[1]}, triggering episode end");
                    }
                }
                else if (cmd == "RL_INIT")
                {
                    tickCount = 0;
                    episodeState = 0;
                    _prevPosValid = false;
                    _pathabilityComputed = false;
                    _pathabilityGrid = null;
                    _eventQueue.Clear();
                    _cdTracker.Clear();
                    _faireRequests.Clear();
                    JassStateCache.Reset();
                    // NOTE: Do NOT reset _grailRegistered here!
                    // RL_INIT fires AFTER hero/grail registration (from RL_WaitReady),
                    // so resetting here would wipe valid grail handles.
                    for (int i = 0; i < MAX_PLAYERS; i++)
                    {
                        _advCount[i] = 0;
                        _faireUsed[i] = 0;
                        _alarmState[i] = false;
                        _prevX[i] = 0f;
                        _prevY[i] = 0f;
                    }
                    if (_speedMultiplier > 1.0)
                    {
                        ApplySyncDelay(10);
                    }
                    // Override local player slot if configured
                    if (LOCAL_PLAYER_SLOT_OVERRIDE >= 0)
                    {
                        PatchLocalPlayerSlot(LOCAL_PLAYER_SLOT_OVERRIDE);
                    }
                    Log($"[RLComm] RL initialized ({heroCount} heroes, speed={_speedMultiplier}x)");
                }
                // ---- Speed scan commands ----
                else if (cmd == "RL_SCAN1")
                {
                    IntPtr gameDll = Kernel32.GetModuleHandleA("game.dll");
                    if (gameDll != IntPtr.Zero)
                        ScanPhase1(gameDll);
                    else
                        Log("[RLComm] SCAN1: game.dll not found");
                }
                else if (cmd == "RL_SCAN2")
                {
                    ScanPhase2();
                }
                else if (cmd == "RL_SPEEDPATCH")
                {
                    PatchTurnInterval();
                }
                else if (cmd == "RL_SCANDUMP")
                {
                    DumpScanState();
                }
                else if (cmd == "RL_DIFFSCAN_PRE")
                {
                    DiffScanPre();
                }
                else if (cmd == "RL_DIFFSCAN_POST")
                {
                    DiffScanPost();
                }
                else if (cmd == "RL_SPEEDTABLE")
                {
                    ReadSpeedTable();
                }
                else if (cmd == "RL_SPEEDTABLE_PATCH")
                {
                    PatchSpeedTable();
                }
            }
            catch (Exception ex)
            {
                Log($"[RLComm] PreloaderOverride error: {ex.Message}");
            }
        }

        // (ReadLine removed -- replaced by UDP binary protocol)

        // ============================================================
        // Grid Computation
        // ============================================================

        /// <summary>
        /// Compute pathability grid (48x25). Cached once on first tick.
        /// Cell value: 1=pathable (walkable), 0=unpathable
        /// IsTerrainPathable returns true if NOT walkable, so we invert.
        /// </summary>
        private static int[] ComputePathabilityGrid()
        {
            int[] grid = new int[GRID_W * GRID_H];
            // ConvertPathingType(2) = WALKABILITY
            var walkable = Natives.ConvertPathingType(2);
            for (int gy = 0; gy < GRID_H; gy++)
            {
                for (int gx = 0; gx < GRID_W; gx++)
                {
                    float wx = MAP_MIN_X + (gx + 0.5f) * CELL_SIZE;
                    float wy = MAP_MIN_Y + (gy + 0.5f) * CELL_SIZE;
                    bool pathable = !Natives.IsTerrainPathable(wx, wy, walkable);
                    grid[gy * GRID_W + gx] = pathable ? 1 : 0;
                }
            }
            return grid;
        }

        /// <summary>
        /// Compute visibility grid for a team (48x25).
        /// Cell value: 1=visible, 0=fogged
        /// </summary>
        private static int[] ComputeVisibilityGrid(int teamStartPid)
        {
            int[] grid = new int[GRID_W * GRID_H];
            // Use first alive player of the team for visibility check
            JassPlayer checkPlayer = default(JassPlayer);
            for (int p = teamStartPid; p < teamStartPid + 6; p++)
            {
                if (heroRegistered[p])
                {
                    checkPlayer = Natives.Player(p);
                    break;
                }
            }
            if (checkPlayer.Handle == IntPtr.Zero) return grid;

            for (int gy = 0; gy < GRID_H; gy++)
            {
                for (int gx = 0; gx < GRID_W; gx++)
                {
                    float wx = MAP_MIN_X + (gx + 0.5f) * CELL_SIZE;
                    float wy = MAP_MIN_Y + (gy + 0.5f) * CELL_SIZE;
                    bool vis = Natives.IsVisibleToPlayer(wx, wy, checkPlayer);
                    grid[gy * GRID_W + gx] = vis ? 1 : 0;
                }
            }
            return grid;
        }

        // Vision ranges (WC3 standard hero sight)
        private const float VISION_DAY = 1800f;
        private const float VISION_NIGHT = 800f;
        // Ward sight is typically smaller; use 1600 day / 800 night
        private const float WARD_VISION_DAY = 1600f;
        private const float WARD_VISION_NIGHT = 800f;
        private const int MAX_VISION_SOURCES = 64; // heroes + wards + summons

        // Static buffers for vision source positions (avoid GC)
        private static float[] _visSrcX = new float[MAX_VISION_SOURCES];
        private static float[] _visSrcY = new float[MAX_VISION_SOURCES];
        private static float[] _visSrcRangeSq = new float[MAX_VISION_SOURCES];
        private static JassGroup _visGroup = default(JassGroup);

        /// <summary>
        /// Math-based visibility: collects ALL alive units per team (heroes + wards + summons)
        /// and computes circular vision from each unit's position.
        /// Avoids IsVisibleToPlayer which leaks WC3 internal memory (~35 bytes/call).
        /// Uses only GetUnitX/Y and GroupEnumUnitsOfPlayer (no leak).
        /// </summary>
        private static void ComputeVisibilityGridInPlace(int teamStartPid, int[] grid)
        {
            Array.Clear(grid, 0, grid.Length);

            // Determine vision range based on time of day
            float timeOfDay = 0f;
            try { timeOfDay = Natives.GetFloatGameState(Natives.ConvertFGameState(2)); } catch { }
            bool isNight = timeOfDay >= 18.0f || timeOfDay < 6.0f;
            float heroRange = isNight ? VISION_NIGHT : VISION_DAY;
            float heroRangeSq = heroRange * heroRange;
            float wardRange = isNight ? WARD_VISION_NIGHT : WARD_VISION_DAY;
            float wardRangeSq = wardRange * wardRange;

            int srcCount = 0;

            // 1. Collect hero positions (fast path, already tracked)
            for (int p = teamStartPid; p < teamStartPid + 6; p++)
            {
                if (!heroRegistered[p]) continue;
                float hp = 0f;
                try { hp = Natives.GetUnitState(heroes[p], JassUnitState.Life); } catch { }
                if (hp <= 0.405f) continue;
                if (srcCount >= MAX_VISION_SOURCES) break;
                try { _visSrcX[srcCount] = Natives.GetUnitX(heroes[p]); } catch { }
                try { _visSrcY[srcCount] = Natives.GetUnitY(heroes[p]); } catch { }
                _visSrcRangeSq[srcCount] = heroRangeSq;
                srcCount++;
            }

            // 2. Collect non-hero units (wards, summons) via GroupEnumUnitsOfPlayer
            //    Set VIS_HERO_ONLY=1 to disable (for leak testing)
            bool heroOnly = Environment.GetEnvironmentVariable("VIS_HERO_ONLY") == "1";
            if (!heroOnly)
            try
            {
                if (_visGroup.Handle == IntPtr.Zero)
                    _visGroup = Natives.CreateGroup();

                for (int p = teamStartPid; p < teamStartPid + 6; p++)
                {
                    if (!heroRegistered[p]) continue;
                    JassPlayer player = Natives.Player(p);
                    Natives.GroupEnumUnitsOfPlayer(_visGroup, player, default(JassBooleanExpression));

                    while (true)
                    {
                        JassUnit u = Natives.FirstOfGroup(_visGroup);
                        if (u.Handle == IntPtr.Zero) break;
                        Natives.GroupRemoveUnit(_visGroup, u);

                        // Skip heroes (already added above)
                        bool isHero = false;
                        for (int h = teamStartPid; h < teamStartPid + 6; h++)
                        {
                            if (heroRegistered[h] && heroes[h].Handle == u.Handle) { isHero = true; break; }
                        }
                        if (isHero) continue;

                        // Check alive (hp > 0)
                        float hp = 0f;
                        try { hp = Natives.GetUnitState(u, JassUnitState.Life); } catch { }
                        if (hp <= 0.405f) continue;

                        if (srcCount >= MAX_VISION_SOURCES) break;
                        try { _visSrcX[srcCount] = Natives.GetUnitX(u); } catch { }
                        try { _visSrcY[srcCount] = Natives.GetUnitY(u); } catch { }
                        _visSrcRangeSq[srcCount] = wardRangeSq;
                        srcCount++;
                    }
                }
            }
            catch { } // GroupEnum failure is non-fatal; heroes still provide vision

            if (srcCount == 0) return;

            // 3. For each grid cell, check distance to ALL vision sources
            for (int gy = 0; gy < GRID_H; gy++)
            {
                float wy = MAP_MIN_Y + (gy + 0.5f) * CELL_SIZE;
                for (int gx = 0; gx < GRID_W; gx++)
                {
                    float wx = MAP_MIN_X + (gx + 0.5f) * CELL_SIZE;
                    for (int s = 0; s < srcCount; s++)
                    {
                        float dx = wx - _visSrcX[s];
                        float dy = wy - _visSrcY[s];
                        if (dx * dx + dy * dy <= _visSrcRangeSq[s])
                        {
                            grid[gy * GRID_W + gx] = 1;
                            break;
                        }
                    }
                }
            }
        }

        // ============================================================
        // JSON State Builder (COMPLETE REWRITE)
        // ============================================================

        /// <summary>
        /// Build the complete JSON state for all heroes and global info.
        /// Called from OnTick (JASS thread).
        /// Also updates episodeState based on team wipe / score detection.
        /// </summary>
        private static string BuildStateJson()
        {
            float gameTime = tickCount * TICK_INTERVAL;
            int team0Alive = 0;
            int team1Alive = 0;

            // Get time of day
            float timeOfDay = 0f;
            try { timeOfDay = Natives.GetFloatGameState(Natives.ConvertFGameState(2)); } // 2 = TIME_OF_DAY
            catch { }
            bool isNight = timeOfDay >= 18.0f || timeOfDay < 6.0f;
            float nextPointTime = 600f - (gameTime % 600f);
            if (nextPointTime > 599.9f) nextPointTime = 600f;

            var unitsArray = new JArray();

            // Current positions array for velocity computation
            float[] curX = new float[MAX_PLAYERS];
            float[] curY = new float[MAX_PLAYERS];

            // Visibility check: we need players for IsUnitVisible
            JassPlayer[] players = new JassPlayer[MAX_PLAYERS];
            for (int i = 0; i < MAX_PLAYERS; i++)
            {
                try { players[i] = Natives.Player(i); } catch { }
            }

            for (int i = 0; i < MAX_PLAYERS; i++)
            {
                if (!heroRegistered[i])
                {
                    var emptyUnit = BuildEmptyUnitJson(i);
                    unitsArray.Add(emptyUnit);
                    continue;
                }

                JassUnit u = heroes[i];
                int typeId = heroTypeIds[i];  // use registered typeId (morph-proof)
                string heroIdStr = TypeIdToString(typeId);

                // Basic stats
                float hp = 0f, maxHp = 0f, mp = 0f, maxMp = 0f;
                float x = 0f, y = 0f;
                bool alive = false;
                int lv = 0;
                float ms = 0f;
                int str = 0, agi = 0, intel = 0;
                float facing = 0f;

                try { hp = Natives.GetUnitState(u, JassUnitState.Life); } catch { }
                try { maxHp = Natives.GetUnitState(u, JassUnitState.MaxLife); } catch { }
                try { mp = Natives.GetUnitState(u, JassUnitState.Mana); } catch { }
                try { maxMp = Natives.GetUnitState(u, JassUnitState.MaxMana); } catch { }
                try { x = Natives.GetUnitX(u); } catch { }
                try { y = Natives.GetUnitY(u); } catch { }
                alive = hp > 0.405f;
                try { lv = (int)Natives.GetHeroLevel(u); } catch { }
                try { ms = Natives.GetUnitMoveSpeed(u); } catch { }
                try { str = (int)Natives.GetHeroStr(u, true); } catch { }
                try { agi = (int)Natives.GetHeroAgi(u, true); } catch { }
                try { intel = (int)Natives.GetHeroInt(u, true); } catch { }
                try { facing = Natives.GetUnitFacing(u); } catch { }

                curX[i] = x;
                curY[i] = y;

                // Velocity
                float velX = 0f, velY = 0f;
                if (_prevPosValid)
                {
                    velX = (x - _prevX[i]) / TICK_INTERVAL;
                    velY = (y - _prevY[i]) / TICK_INTERVAL;
                }

                // XP, skill points, stat points
                int xp = 0;
                int skillPoints = 0;
                int statPoints = 0;
                try { xp = (int)Natives.GetHeroXP(u); } catch { }
                try { skillPoints = (int)Natives.GetHeroSkillPoints(u); } catch { }
                statPoints = GetStatPoints(i);

                // Attack, defense, attack speed, range
                float atk = 0f, def = 0f, atkRange = 128f, atkSpd = 1.7f;
                int mainStat = 0;
                HeroData hdata;
                if (_heroDataTable.TryGetValue(typeId, out hdata))
                {
                    mainStat = hdata.mainStat;
                    int mainStatVal = mainStat == 0 ? str : mainStat == 1 ? agi : intel;
                    atk = hdata.baseAtk + mainStatVal * hdata.atkPerStr;
                    def = hdata.baseDef;
                    atkRange = hdata.atkRange;
                    atkSpd = hdata.baseAtkSpd;
                }

                // Skills
                var skillsObj = BuildSkillsJson(i, u, typeId, lv, gameTime);

                // Upgrades from JASS cache
                var upgradesArr = new JArray();
                for (int ui = 0; ui < 9; ui++)
                    upgradesArr.Add(JassStateCache.upgrades[i, ui]);

                // Attributes: derived from attrCount (sequential acquisition: first, second, third, fourth)
                int attrCount = JassStateCache.attrCount[i];
                var attrArr = new JArray(attrCount >= 1, attrCount >= 2, attrCount >= 3, attrCount >= 4);

                // Buffs detection via GetUnitAbilityLevel
                bool isStunned = false, isSlowed = false, isSilenced = false;
                bool isKnockback = false, isRooted = false, isInvuln = false;
                try { isStunned = (int)Natives.GetUnitAbilityLevel(u, (JassObjectId)_buffStun) > 0; } catch { }
                try { isSlowed = (int)Natives.GetUnitAbilityLevel(u, (JassObjectId)_buffSlow) > 0; } catch { }
                try { isSilenced = (int)Natives.GetUnitAbilityLevel(u, (JassObjectId)_buffSilence) > 0; } catch { }
                try { isRooted = (int)Natives.GetUnitAbilityLevel(u, (JassObjectId)_buffRoot) > 0; } catch { }
                try { isInvuln = (int)Natives.GetUnitAbilityLevel(u, (JassObjectId)_buffInvuln) > 0; } catch { }

                var buffsObj = new JObject
                {
                    ["stun"] = isStunned,
                    ["slow"] = isSlowed,
                    ["silence"] = isSilenced,
                    ["knockback"] = isKnockback,
                    ["root"] = isRooted,
                    ["invuln"] = isInvuln
                };

                // Seal data
                var sealData = GetSealItem(i);
                int sealCharges = sealData.charges;

                // Items
                var itemsArr = BuildItemsJson(u);

                // Gold (faire)
                int gold = GetPlayerGold(i);

                // Visibility: is this unit visible to the ENEMY team?
                bool visibleToEnemy = true;
                int enemyCheckPid = i < 6 ? 6 : 0;
                try
                {
                    if (players[enemyCheckPid].Handle != IntPtr.Zero)
                        visibleToEnemy = Natives.IsUnitVisible(u, players[enemyCheckPid]);
                }
                catch { }

                // Revive remain placeholder (would need JASS timer tracking)
                float reviveRemain = alive ? 0f : 10f; // approximate: 10s dead time

                // Track alive counts
                if (alive)
                {
                    if (i < 6) team0Alive++;
                    else team1Alive++;
                }

                // Build action mask
                var actionMask = BuildActionMask(i, u, typeId, lv, alive, mp, gold, skillPoints, statPoints, gameTime);

                var unitObj = new JObject
                {
                    ["idx"] = i,
                    ["hero_id"] = heroIdStr,
                    ["team"] = i < 6 ? 0 : 1,
                    // basic
                    ["hp"] = (float)Math.Round(hp, 1),
                    ["max_hp"] = (float)Math.Round(maxHp, 1),
                    ["mp"] = (float)Math.Round(mp, 1),
                    ["max_mp"] = (float)Math.Round(maxMp, 1),
                    ["x"] = (float)Math.Round(x, 1),
                    ["y"] = (float)Math.Round(y, 1),
                    ["vel_x"] = (float)Math.Round(velX, 1),
                    ["vel_y"] = (float)Math.Round(velY, 1),
                    ["alive"] = alive,
                    ["revive_remain"] = (float)Math.Round(reviveRemain, 1),
                    // stats
                    ["str"] = str,
                    ["agi"] = agi,
                    ["int"] = intel,
                    ["atk"] = (float)Math.Round(atk, 1),
                    ["def"] = (float)Math.Round(def, 1),
                    ["move_spd"] = (float)Math.Round(ms, 1),
                    ["atk_range"] = atkRange,
                    ["atk_spd"] = atkSpd,
                    // growth
                    ["level"] = lv,
                    ["xp"] = xp,
                    ["skill_points"] = skillPoints,
                    ["stat_points"] = statPoints,
                    // skills
                    ["skills"] = skillsObj,
                    // upgrades
                    ["upgrades"] = upgradesArr,
                    // attributes
                    ["attributes"] = attrArr,
                    // buffs
                    ["buffs"] = buffsObj,
                    // seal
                    ["seal_charges"] = sealCharges,
                    ["seal_cd"] = JassStateCache.sealCd[i],
                    ["seal_first_active"] = JassStateCache.firstActive[i] != 0,
                    ["seal_first_remain"] = JassStateCache.firstRemain[i],
                    // items
                    ["items"] = itemsArr,
                    // economy
                    ["faire"] = gold,
                    ["faire_cap"] = FAIRE_CAP,
                    // alarm
                    ["enemy_alarm"] = _alarmState[i],
                    // visibility (to enemy)
                    ["visible"] = visibleToEnemy,
                    // action mask
                    ["action_mask"] = actionMask,
                };

                // Visibility masking: if this unit is an enemy and NOT visible, zero out most fields
                // We apply this per-observer, but since we send a global state, we keep full data
                // and let the Python side handle fog-of-war masking per team perspective.
                // However, we include the visible flag so Python can mask as needed.

                unitsArray.Add(unitObj);
            }

            // Update previous positions
            for (int i = 0; i < MAX_PLAYERS; i++)
            {
                _prevX[i] = curX[i];
                _prevY[i] = curY[i];
            }
            _prevPosValid = true;

            // Episode state: score-based only (no team wipe — heroes respawn)
            if (episodeState == 0)
            {
                if (JassStateCache.teamScore[0] >= TARGET_SCORE)
                {
                    episodeState = 4;
                }
                else if (JassStateCache.teamScore[1] >= TARGET_SCORE)
                {
                    episodeState = 4;
                }
            }

            // Events: flush and include
            var eventsArr = _eventQueue.FlushAndClear();

            // Shop info
            var shopObj = new JObject
            {
                ["c_rank_stock"] = 0  // deprecated, kept for compatibility
            };

            // Global
            var globalObj = new JObject
            {
                ["game_time"] = (float)Math.Round(gameTime, 1),
                ["time_of_day"] = (float)Math.Round(timeOfDay, 1),
                ["is_night"] = isNight,
                ["next_point_time"] = (float)Math.Round(nextPointTime, 1),
                ["score_ally"] = JassStateCache.teamScore[0],
                ["score_enemy"] = JassStateCache.teamScore[1],
                ["target_score"] = TARGET_SCORE
            };

            // Build grids section
            var gridsObj = new JObject();

            // Pathability: computed once on first tick
            if (!_pathabilityComputed)
            {
                try
                {
                    _pathabilityGrid = ComputePathabilityGrid();
                    _pathabilityComputed = true;
                    Log("[RLComm] Pathability grid computed");
                }
                catch (Exception ex)
                {
                    Log($"[RLComm] Pathability grid error: {ex.Message}");
                    _pathabilityGrid = new int[GRID_W * GRID_H]; // all zeros
                    _pathabilityComputed = true;
                }
            }

            // Only send pathability on first tick to reduce bandwidth
            if (tickCount == 1 && _pathabilityGrid != null)
            {
                gridsObj["pathability"] = new JArray(_pathabilityGrid);
            }

            // Visibility grids (every tick)
            try
            {
                int[] visTeam0 = ComputeVisibilityGrid(0);
                int[] visTeam1 = ComputeVisibilityGrid(6);
                gridsObj["visibility_team0"] = new JArray(visTeam0);
                gridsObj["visibility_team1"] = new JArray(visTeam1);
            }
            catch (Exception ex)
            {
                Log($"[RLComm] Visibility grid error: {ex.Message}");
            }

            var stateObj = new JObject
            {
                ["tick"] = tickCount,
                ["global"] = globalObj,
                ["units"] = unitsArray,
                ["shop"] = shopObj,
                ["events"] = eventsArr,
                ["grids"] = gridsObj
            };

            return stateObj.ToString(Formatting.None);
        }

        private static JObject BuildEmptyUnitJson(int idx)
        {
            var emptySkills = new JObject();
            string[] slotNames = { "Q", "W", "E", "R", "D", "F" };
            foreach (string sn in slotNames)
            {
                emptySkills[sn] = new JObject
                {
                    ["abil_id"] = 0, ["level"] = 0, ["cd_remain"] = 0f, ["cd_max"] = 0f, ["exists"] = false
                };
            }

            var emptyItems = new JArray();
            for (int s = 0; s < 6; s++)
            {
                emptyItems.Add(new JObject { ["slot"] = s, ["type_id"] = null, ["charges"] = 0 });
            }

            var emptyUpgrades = new JArray(0, 0, 0, 0, 0, 0, 0, 0, 0);
            var emptyMask = BuildEmptyActionMask();

            return new JObject
            {
                ["idx"] = idx,
                ["hero_id"] = "0000",
                ["team"] = idx < 6 ? 0 : 1,
                ["hp"] = 0f, ["max_hp"] = 0f, ["mp"] = 0f, ["max_mp"] = 0f,
                ["x"] = 0f, ["y"] = 0f, ["vel_x"] = 0f, ["vel_y"] = 0f,
                ["alive"] = false, ["revive_remain"] = 0f,
                ["str"] = 0, ["agi"] = 0, ["int"] = 0,
                ["atk"] = 0f, ["def"] = 0f, ["move_spd"] = 0f,
                ["atk_range"] = 0f, ["atk_spd"] = 0f,
                ["level"] = 0, ["xp"] = 0, ["skill_points"] = 0, ["stat_points"] = 0,
                ["skills"] = emptySkills,
                ["upgrades"] = emptyUpgrades,
                ["attributes"] = new JArray(false, false, false, false),
                ["buffs"] = new JObject
                {
                    ["stun"] = false, ["slow"] = false, ["silence"] = false,
                    ["knockback"] = false, ["root"] = false, ["invuln"] = false
                },
                ["seal_charges"] = 0, ["seal_cd"] = 0,
                ["seal_first_active"] = false, ["seal_first_remain"] = 0,
                ["items"] = emptyItems,
                ["faire"] = 0, ["faire_cap"] = FAIRE_CAP,
                ["enemy_alarm"] = false,
                ["visible"] = false,
                ["action_mask"] = emptyMask,
            };
        }

        private static JObject BuildSkillsJson(int heroIdx, JassUnit u, int typeId, int heroLevel, float gameTime)
        {
            var skillsObj = new JObject();
            string[] slotNames = { "Q", "W", "E", "R", "D", "F" };

            HeroData hdata;
            bool hasData = _heroDataTable.TryGetValue(typeId, out hdata);

            for (int s = 0; s < 6; s++)
            {
                int abilLevel = 0;
                float cdRemain = 0f;
                float cdMax = 0f;
                int abilId = 0;
                bool exists = false;

                if (hasData && s < hdata.skills.Length && hdata.skills[s].abilId != 0)
                {
                    SkillInfo si = hdata.skills[s];
                    abilId = si.abilId;
                    exists = true;
                    try { abilLevel = (int)Natives.GetUnitAbilityLevel(u, (JassObjectId)si.abilId); } catch { }
                    cdRemain = _cdTracker.GetCdRemain(heroIdx, s, gameTime);
                    if (abilLevel > 0 && abilLevel <= si.maxCd.Length)
                        cdMax = si.maxCd[abilLevel - 1];
                }

                skillsObj[slotNames[s]] = new JObject
                {
                    ["abil_id"] = abilId,
                    ["level"] = abilLevel,
                    ["cd_remain"] = (float)Math.Round(cdRemain, 1),
                    ["cd_max"] = cdMax,
                    ["exists"] = exists
                };
            }

            return skillsObj;
        }

        private static JArray BuildItemsJson(JassUnit u)
        {
            var itemsArr = new JArray();
            for (int slot = 0; slot < 6; slot++)
            {
                try
                {
                    JassItem itm = Natives.UnitItemInSlot(u, slot);
                    if (itm.Handle != IntPtr.Zero)
                    {
                        int itmType = (int)Natives.GetItemTypeId(itm);
                        int charges = (int)Natives.GetItemCharges(itm);
                        string itmTypeStr = TypeIdToString(itmType);

                        // Skip "null" items (type == 0)
                        if (itmType == 0)
                        {
                            itemsArr.Add(new JObject { ["slot"] = slot, ["type_id"] = null, ["charges"] = 0 });
                        }
                        else
                        {
                            itemsArr.Add(new JObject { ["slot"] = slot, ["type_id"] = itmTypeStr, ["charges"] = charges });
                        }
                    }
                    else
                    {
                        itemsArr.Add(new JObject { ["slot"] = slot, ["type_id"] = null, ["charges"] = 0 });
                    }
                }
                catch
                {
                    itemsArr.Add(new JObject { ["slot"] = slot, ["type_id"] = null, ["charges"] = 0 });
                }
            }
            return itemsArr;
        }

        // ============================================================
        // Action Masking
        // ============================================================

        private static JObject BuildEmptyActionMask()
        {
            return new JObject
            {
                ["skill"] = new JArray(true, false, false, false, false, false, false, false),
                ["unit_target"] = new JArray(false, false, false, false, false, false, false, false, false, false, false, false, true, true),
                ["skill_levelup"] = new JArray(true, false, false, false, false, false),
                ["stat_upgrade"] = new JArray(true, false, false, false, false, false, false, false, false, false),
                ["attribute"] = new JArray(true, false, false, false, false),
                ["item_buy"] = BuildFalseArrayWithNone(17),
                ["item_use"] = BuildFalseArrayWithNone(7),
                ["seal_use"] = BuildFalseArrayWithNone(7),
                ["faire_send"] = BuildFalseArrayWithNone(6),
                ["faire_request"] = BuildFalseArrayWithNone(6),
                ["faire_respond"] = BuildFalseArrayWithNone(3),
            };
        }

        /// <summary>Build array of booleans: first=true (none option), rest=false</summary>
        private static JArray BuildFalseArrayWithNone(int size)
        {
            var arr = new JArray();
            arr.Add(true); // index 0 = none, always available
            for (int i = 1; i < size; i++) arr.Add(false);
            return arr;
        }

        private static JObject BuildActionMask(int heroIdx, JassUnit u, int typeId, int heroLevel,
            bool alive, float mp, int gold, int skillPoints, int statPoints, float gameTime)
        {
            // If dead, all masks are false except "none" options (index 0)
            // Exception: seal_use[5]=revive may be available
            if (!alive)
            {
                var deadMask = BuildEmptyActionMask();

                // Check if revive seal is available (seal_use[5])
                var sealData = GetSealItem(heroIdx);
                int sealCharges = sealData.charges;
                bool firstActive = JassStateCache.firstActive[heroIdx] != 0;
                int reviveCost = firstActive ? _sealCostsFirstActive[5] : _sealCosts[5];
                if (sealCharges >= reviveCost && JassStateCache.sealCd[heroIdx] <= 0)
                {
                    ((JArray)deadMask["seal_use"])[5] = true;
                }

                return deadMask;
            }

            // ---- Skill mask (size 8): 0=none,1=attack,2=Q,3=W,4=E,5=R,6=D,7=F ----
            var skillMask = new JArray();
            skillMask.Add(true); // 0=none always available
            skillMask.Add(alive); // 1=attack available when alive

            HeroData hdata;
            bool hasData = _heroDataTable.TryGetValue(typeId, out hdata);

            string _dbgSkillMask = "";
            for (int s = 0; s < 6; s++) // Q,W,E,R,D,F
            {
                bool canUse = false;
                if (alive && hasData && s < hdata.skills.Length && hdata.skills[s].abilId != 0
                    && !string.IsNullOrEmpty(hdata.skills[s].orderId))
                {
                    SkillInfo si = hdata.skills[s];
                    int abilLevel = 0;
                    try { abilLevel = (int)Natives.GetUnitAbilityLevel(u, (JassObjectId)si.abilId); } catch { }
                    if (abilLevel > 0)
                    {
                        float cdRemain = _cdTracker.GetCdRemain(heroIdx, s, gameTime);
                        int manaCost = 0;
                        if (abilLevel <= si.manaCost.Length)
                            manaCost = si.manaCost[abilLevel - 1];
                        canUse = cdRemain <= 0f && mp >= manaCost;
                    }
                    _dbgSkillMask += $" {s}:lv{abilLevel}/mc{(abilLevel > 0 && abilLevel <= si.manaCost.Length ? si.manaCost[abilLevel-1] : 0)}/{(canUse?"Y":"N")}";
                }
                skillMask.Add(canUse);
            }
            // Periodic skill mask debug log (every ~200 calls per hero)
            if (hasData && _maskDebugCount[heroIdx]++ % 200 == 0)
            {
                float maxMp = 0;
                try { maxMp = Natives.GetUnitState(u, JassUnitState.MaxMana); } catch {}
                Log($"[SkillMask] P{heroIdx} {hdata.heroId} mp={mp:F0}/{maxMp:F0} lv={heroLevel}{_dbgSkillMask}");
            }

            // ---- Unit target mask (size 14): 0-5=ally, 6-11=enemy, 12=self, 13=point_mode ----
            var unitTargetMask = new JArray();
            int myTeamBase = heroIdx < 6 ? 0 : 6;
            int enemyTeamBase = heroIdx < 6 ? 6 : 0;

            for (int t = 0; t < 14; t++)
            {
                if (t < 6)
                {
                    // Ally indices 0-5
                    int allyPid = myTeamBase + t;
                    bool canTarget = heroRegistered[allyPid] && IsUnitAlive(heroes[allyPid]);
                    unitTargetMask.Add(canTarget);
                }
                else if (t < 12)
                {
                    // Enemy indices 6-11
                    int enemyPid = enemyTeamBase + (t - 6);
                    bool canTarget = false;
                    if (heroRegistered[enemyPid] && IsUnitAlive(heroes[enemyPid]))
                    {
                        // Enemy must be visible
                        try
                        {
                            JassPlayer myPlayer = Natives.Player(heroIdx);
                            canTarget = Natives.IsUnitVisible(heroes[enemyPid], myPlayer);
                        }
                        catch { }
                    }
                    unitTargetMask.Add(canTarget);
                }
                else if (t == 12)
                {
                    unitTargetMask.Add(true); // self always available
                }
                else // t == 13
                {
                    unitTargetMask.Add(true); // point_mode always available
                }
            }

            // ---- Skill levelup mask (size 6): 0=none, 1=Q, 2=W, 3=E, 4=R, 5=allstat ----
            var skillLevelupMask = new JArray();
            skillLevelupMask.Add(true); // 0=none
            bool allQwerMaxed = true;
            for (int s = 0; s < 4; s++) // Q,W,E,R
            {
                bool canLevel = false;
                if (skillPoints > 0 && hasData && s < hdata.skills.Length && hdata.skills[s].abilId != 0)
                {
                    SkillInfo si = hdata.skills[s];
                    int checkAbilId = si.learnAbilId != 0 ? si.learnAbilId : si.abilId;
                    int curLevel = 0;
                    try { curLevel = (int)Natives.GetUnitAbilityLevel(u, (JassObjectId)checkAbilId); } catch { }
                    int maxLevel = s == 3 ? 3 : 5; // R usually max 3, others max 5
                    canLevel = curLevel < maxLevel;
                    if (curLevel < 5) allQwerMaxed = false;
                }
                else
                {
                    allQwerMaxed = false;
                }
                skillLevelupMask.Add(skillPoints > 0 && canLevel);
            }
            // 5=allstat: requires skill_points>0 AND all QWER at level 5
            skillLevelupMask.Add(skillPoints > 0 && allQwerMaxed);

            // ---- Stat upgrade mask (size 10): 0=none, 1-9=stats ----
            var statUpgradeMask = new JArray();
            statUpgradeMask.Add(true); // 0=none
            for (int s = 0; s < 9; s++)
            {
                bool canUpgrade = statPoints > 0 && JassStateCache.upgrades[heroIdx, s] < STAT_UPGRADE_CAPS[s];
                statUpgradeMask.Add(canUpgrade);
            }

            // ---- Attribute mask (size 5): 0=none, 1-4=A/B/C/D ----
            // No sequential constraint — any attribute can be chosen if not yet acquired and affordable
            // Lancelot (H03M) attr4 can be acquired twice
            bool isLancelot = hasData && hdata.heroId == "H03M";
            var attrMask = new JArray();
            attrMask.Add(true); // 0=none
            for (int a = 0; a < 4; a++)
            {
                int maxAcq = (isLancelot && a == 3) ? 2 : 1;
                bool canAcquire = JassStateCache.attrAcquired[heroIdx, a + 1] < maxAcq;
                if (canAcquire && hasData && hdata.attributeCost != null && a < hdata.attributeCost.Length)
                    canAcquire = statPoints >= hdata.attributeCost[a];
                else if (canAcquire)
                    canAcquire = statPoints >= 10; // default cost
                attrMask.Add(canAcquire);
            }

            // ---- Item buy mask (size 18): 0=none, 1-6=faire options, 7-17=items ----
            bool hasSlot = HasEmptyItemSlot(u);
            var itemBuyMask = new JArray();
            itemBuyMask.Add(true); // 0=none
            // idx 1-6: faire options (available when advCount/7 > faireUsed)
            bool faireAvailable = (_advCount[heroIdx] / 7) > _faireUsed[heroIdx];
            for (int bi = 1; bi <= 6; bi++)
            {
                itemBuyMask.Add(faireAvailable && hasSlot);
            }
            for (int bi = 7; bi <= 17; bi++)
            {
                bool canBuy = hasSlot && bi < _shopItems.Length && gold >= _shopItems[bi].cost;
                itemBuyMask.Add(canBuy);
            }

            // ---- Item use mask (size 7): 0=none, 1-6=slot ----
            var itemUseMask = new JArray();
            itemUseMask.Add(true); // 0=none
            for (int slot = 0; slot < 6; slot++)
            {
                bool hasItem = false;
                try
                {
                    JassItem itm = Natives.UnitItemInSlot(u, slot);
                    if (itm.Handle != IntPtr.Zero)
                    {
                        int itmType = (int)Natives.GetItemTypeId(itm);
                        hasItem = itmType != 0;
                    }
                }
                catch { }
                itemUseMask.Add(alive && hasItem);
            }

            // ---- Seal use mask (size 7): 0=none, 1=first_activate, 2=cd_reset, 3=hp_recover, 4=mp_recover, 5=revive, 6=teleport ----
            var sealUseMask = new JArray();
            sealUseMask.Add(true); // 0=none
            var sealInfo = GetSealItem(heroIdx);
            int sealChg = sealInfo.charges;
            bool isFirstActive = JassStateCache.firstActive[heroIdx] != 0;
            int sealCd = JassStateCache.sealCd[heroIdx];

            for (int si = 1; si <= 6; si++)
            {
                int cost = isFirstActive ? _sealCostsFirstActive[si] : _sealCosts[si];
                bool canUse = sealChg >= cost && sealCd <= 0;

                // Special conditions:
                // 1 = first seal activate: only if not already active
                if (si == 1) canUse = canUse && !isFirstActive;
                // 5 = revive: usable even when dead (handled above in dead mask)
                // When alive, revive makes no sense
                if (si == 5 && alive) canUse = false;

                sealUseMask.Add(canUse);
            }

            // ---- Faire send mask (size 6): 0=none, 1-5=ally (same team, excluding self) ----
            var faireSendMask = new JArray();
            faireSendMask.Add(true); // 0=none
            int allyCount = 0;
            for (int a = myTeamBase; a < myTeamBase + 6; a++)
            {
                if (a == heroIdx) continue;
                allyCount++;
                if (allyCount > 5) break;
                bool canSend = gold >= FAIRE_TRANSFER_AMOUNT && heroRegistered[a] && IsUnitAlive(heroes[a]);
                faireSendMask.Add(canSend);
            }
            // Pad if needed
            while (faireSendMask.Count < 6) faireSendMask.Add(false);

            // ---- Faire request mask (size 6): 0=none, 1-5=ally ----
            var faireRequestMask = new JArray();
            faireRequestMask.Add(true); // 0=none
            allyCount = 0;
            for (int a = myTeamBase; a < myTeamBase + 6; a++)
            {
                if (a == heroIdx) continue;
                allyCount++;
                if (allyCount > 5) break;
                bool canReq = heroRegistered[a] && IsUnitAlive(heroes[a]);
                faireRequestMask.Add(canReq);
            }
            while (faireRequestMask.Count < 6) faireRequestMask.Add(false);

            // ---- Faire respond mask (size 3): 0=none, 1=accept, 2=deny ----
            var faireRespondMask = new JArray();
            faireRespondMask.Add(true); // 0=none
            bool hasPendingRequest = _faireRequests.ContainsKey(heroIdx);
            if (hasPendingRequest)
            {
                var req = _faireRequests[heroIdx];
                bool canAccept = gold >= req.amount;
                faireRespondMask.Add(canAccept);
                faireRespondMask.Add(true); // can always deny
            }
            else
            {
                faireRespondMask.Add(false);
                faireRespondMask.Add(false);
            }

            return new JObject
            {
                ["skill"] = skillMask,
                ["unit_target"] = unitTargetMask,
                ["skill_levelup"] = skillLevelupMask,
                ["stat_upgrade"] = statUpgradeMask,
                ["attribute"] = attrMask,
                ["item_buy"] = itemBuyMask,
                ["item_use"] = itemUseMask,
                ["seal_use"] = sealUseMask,
                ["faire_send"] = faireSendMask,
                ["faire_request"] = faireRequestMask,
                ["faire_respond"] = faireRespondMask,
            };
        }

        // ============================================================
        // ActionMaskBits builders (zero-allocation for binary path)
        // ============================================================

        /// <summary>
        /// Build empty (dead/unregistered) action mask as packed bits.
        /// Only "none" options (index 0) are set, plus unit_target self(12) and point_mode(13).
        /// </summary>
        private static ActionMaskBits BuildEmptyActionMaskBits()
        {
            ActionMaskBits m;
            m.skill = 0x01;             // bit 0 = none
            m.unitTarget = (ushort)((1 << 12) | (1 << 13));  // self + point_mode
            m.skillLevelup = 0x01;      // bit 0 = none
            m.statUpgrade = 0x0001;     // bit 0 = none
            m.attribute = 0x01;         // bit 0 = none
            m.itemBuy = 0x00000001u;    // bit 0 = none
            m.itemUse = 0x01;           // bit 0 = none
            m.sealUse = 0x01;           // bit 0 = none
            m.faireSend = 0x01;         // bit 0 = none
            m.faireRequest = 0x01;      // bit 0 = none
            m.faireRespond = 0x01;      // bit 0 = none
            return m;
        }

        /// <summary>
        /// Build action mask as packed bits (zero-allocation).
        /// Game logic is identical to BuildActionMask (JObject version).
        /// </summary>
        private static ActionMaskBits BuildActionMaskBits(int heroIdx, JassUnit u, int typeId, int heroLevel,
            bool alive, float mp, int gold, int skillPoints, int statPoints, float gameTime)
        {
            // If dead, all masks are false except "none" options (index 0)
            // Exception: seal_use[5]=revive may be available
            if (!alive)
            {
                var deadMask = BuildEmptyActionMaskBits();

                // Check if revive seal is available (seal_use[5])
                var sealData = GetSealItem(heroIdx);
                int sealCharges = sealData.charges;
                bool firstActive = JassStateCache.firstActive[heroIdx] != 0;
                int reviveCost = firstActive ? _sealCostsFirstActive[5] : _sealCosts[5];
                if (sealCharges >= reviveCost && JassStateCache.sealCd[heroIdx] <= 0)
                {
                    deadMask.sealUse |= (byte)(1 << 5);
                }

                return deadMask;
            }

            ActionMaskBits result;

            // ---- Skill mask (size 8): 0=none,1=attack,2=Q,3=W,4=E,5=R,6=D,7=F ----
            result.skill = 0x01; // bit 0 = none always available
            if (alive) result.skill |= (byte)(1 << 1); // bit 1 = attack

            HeroData hdata;
            bool hasData = _heroDataTable.TryGetValue(typeId, out hdata);

            for (int s = 0; s < 6; s++) // Q,W,E,R,D,F
            {
                bool canUse = false;
                if (alive && hasData && s < hdata.skills.Length && hdata.skills[s].abilId != 0
                    && !string.IsNullOrEmpty(hdata.skills[s].orderId))
                {
                    SkillInfo si = hdata.skills[s];
                    int abilLevel = 0;
                    try { abilLevel = (int)Natives.GetUnitAbilityLevel(u, (JassObjectId)si.abilId); } catch { }
                    if (abilLevel > 0)
                    {
                        float cdRemain = _cdTracker.GetCdRemain(heroIdx, s, gameTime);
                        int manaCost = 0;
                        if (abilLevel <= si.manaCost.Length)
                            manaCost = si.manaCost[abilLevel - 1];
                        canUse = cdRemain <= 0f && mp >= manaCost;
                    }
                }
                if (canUse) result.skill |= (byte)(1 << (s + 2));
            }

            // ---- Unit target mask (size 14): 0-5=ally, 6-11=enemy, 12=self, 13=point_mode ----
            result.unitTarget = 0;
            int myTeamBase = heroIdx < 6 ? 0 : 6;
            int enemyTeamBase = heroIdx < 6 ? 6 : 0;

            for (int t = 0; t < 14; t++)
            {
                bool canTarget = false;
                if (t < 6)
                {
                    int allyPid = myTeamBase + t;
                    canTarget = heroRegistered[allyPid] && IsUnitAlive(heroes[allyPid]);
                }
                else if (t < 12)
                {
                    int enemyPid = enemyTeamBase + (t - 6);
                    if (heroRegistered[enemyPid] && IsUnitAlive(heroes[enemyPid]))
                    {
                        if (_fogInitialized)
                        {
                            // Fog bitmap: check if enemy hero's position is visible to me
                            float ex = 0f, ey = 0f;
                            try { ex = Natives.GetUnitX(heroes[enemyPid]); } catch { }
                            try { ey = Natives.GetUnitY(heroes[enemyPid]); } catch { }
                            canTarget = IsUnitVisibleViaFog(ex, ey, heroIdx);
                        }
                        else
                        {
                            try
                            {
                                JassPlayer myPlayer = Natives.Player(heroIdx);
                                canTarget = Natives.IsUnitVisible(heroes[enemyPid], myPlayer);
                            }
                            catch { }
                        }
                    }
                }
                else if (t == 12)
                {
                    canTarget = true; // self always available
                }
                else // t == 13
                {
                    canTarget = true; // point_mode always available
                }
                if (canTarget) result.unitTarget |= (ushort)(1 << t);
            }

            // ---- Skill levelup mask (size 6): 0=none, 1=Q, 2=W, 3=E, 4=R, 5=allstat ----
            result.skillLevelup = 0x01; // bit 0 = none
            bool allQwerMaxed = true;
            for (int s = 0; s < 4; s++) // Q,W,E,R
            {
                bool canLevel = false;
                if (skillPoints > 0 && hasData && s < hdata.skills.Length && hdata.skills[s].abilId != 0)
                {
                    SkillInfo si = hdata.skills[s];
                    int checkAbilId = si.learnAbilId != 0 ? si.learnAbilId : si.abilId;
                    int curLevel = 0;
                    try { curLevel = (int)Natives.GetUnitAbilityLevel(u, (JassObjectId)checkAbilId); } catch { }
                    int maxLevel = s == 3 ? 3 : 5; // R usually max 3, others max 5
                    canLevel = curLevel < maxLevel;
                    if (curLevel < 5) allQwerMaxed = false;
                }
                else
                {
                    allQwerMaxed = false;
                }
                if (skillPoints > 0 && canLevel) result.skillLevelup |= (byte)(1 << (s + 1));
            }
            // 5=allstat: requires skill_points>0 AND all QWER at level 5
            if (skillPoints > 0 && allQwerMaxed) result.skillLevelup |= (byte)(1 << 5);

            // ---- Stat upgrade mask (size 10): 0=none, 1-9=stats ----
            result.statUpgrade = 0x0001; // bit 0 = none
            for (int s = 0; s < 9; s++)
            {
                bool canUpgrade = statPoints > 0 && JassStateCache.upgrades[heroIdx, s] < STAT_UPGRADE_CAPS[s];
                if (canUpgrade) result.statUpgrade |= (ushort)(1 << (s + 1));
            }

            // ---- Attribute mask (size 5): 0=none, 1-4=A/B/C/D ----
            // No sequential constraint — any attribute freely choosable
            // Lancelot (H03M) attr4 can be acquired twice
            bool isLancelotB = hasData && hdata.heroId == "H03M";
            result.attribute = 0x01; // bit 0 = none
            for (int a = 0; a < 4; a++)
            {
                int maxAcq = (isLancelotB && a == 3) ? 2 : 1;
                bool canAcquire = JassStateCache.attrAcquired[heroIdx, a + 1] < maxAcq;
                if (canAcquire && hasData && hdata.attributeCost != null && a < hdata.attributeCost.Length)
                    canAcquire = statPoints >= hdata.attributeCost[a];
                else if (canAcquire)
                    canAcquire = statPoints >= 10;
                if (canAcquire) result.attribute |= (byte)(1 << (a + 1));
            }

            // ---- Item buy mask (size 18): 0=none, 1-6=faire options, 7-17=items ----
            bool hasSlot = HasEmptyItemSlot(u);
            result.itemBuy = 0x00000001u; // bit 0 = none
            // idx 1-6: faire options (available when advCount/7 > faireUsed)
            bool faireAvailableB = (_advCount[heroIdx] / 7) > _faireUsed[heroIdx];
            for (int bi = 1; bi <= 6; bi++)
            {
                if (faireAvailableB && hasSlot) result.itemBuy |= (uint)(1 << bi);
            }
            for (int bi = 7; bi <= 17; bi++)
            {
                bool canBuy = hasSlot && bi < _shopItems.Length && gold >= _shopItems[bi].cost;
                if (canBuy) result.itemBuy |= (uint)(1 << bi);
            }

            // ---- Item use mask (size 7): 0=none, 1-6=slot ----
            result.itemUse = 0x01; // bit 0 = none
            for (int slot = 0; slot < 6; slot++)
            {
                bool hasItem = false;
                try
                {
                    JassItem itm = Natives.UnitItemInSlot(u, slot);
                    if (itm.Handle != IntPtr.Zero)
                    {
                        int itmType = (int)Natives.GetItemTypeId(itm);
                        hasItem = itmType != 0;
                    }
                }
                catch { }
                if (alive && hasItem) result.itemUse |= (byte)(1 << (slot + 1));
            }

            // ---- Seal use mask (size 7): 0=none, 1=first_activate, 2=cd_reset, 3=hp_recover, 4=mp_recover, 5=revive, 6=teleport ----
            result.sealUse = 0x01; // bit 0 = none
            var sealInfo = GetSealItem(heroIdx);
            int sealChg = sealInfo.charges;
            bool isFirstActive = JassStateCache.firstActive[heroIdx] != 0;
            int sealCd = JassStateCache.sealCd[heroIdx];

            for (int si = 1; si <= 6; si++)
            {
                int cost = isFirstActive ? _sealCostsFirstActive[si] : _sealCosts[si];
                bool canUse = sealChg >= cost && sealCd <= 0;

                // Special conditions:
                // 1 = first seal activate: only if not already active
                if (si == 1) canUse = canUse && !isFirstActive;
                // 5 = revive: usable even when dead (handled above in dead mask)
                // When alive, revive makes no sense
                if (si == 5 && alive) canUse = false;

                if (canUse) result.sealUse |= (byte)(1 << si);
            }

            // ---- Faire send mask (size 6): 0=none, 1-5=ally (same team, excluding self) ----
            result.faireSend = 0x01; // bit 0 = none
            int allyCount = 0;
            for (int a = myTeamBase; a < myTeamBase + 6; a++)
            {
                if (a == heroIdx) continue;
                allyCount++;
                if (allyCount > 5) break;
                bool canSend = gold >= FAIRE_TRANSFER_AMOUNT && heroRegistered[a] && IsUnitAlive(heroes[a]);
                if (canSend) result.faireSend |= (byte)(1 << allyCount);
            }

            // ---- Faire request mask (size 6): 0=none, 1-5=ally ----
            result.faireRequest = 0x01; // bit 0 = none
            allyCount = 0;
            for (int a = myTeamBase; a < myTeamBase + 6; a++)
            {
                if (a == heroIdx) continue;
                allyCount++;
                if (allyCount > 5) break;
                bool canReq = heroRegistered[a] && IsUnitAlive(heroes[a]);
                if (canReq) result.faireRequest |= (byte)(1 << allyCount);
            }

            // ---- Faire respond mask (size 3): 0=none, 1=accept, 2=deny ----
            result.faireRespond = 0x01; // bit 0 = none
            bool hasPendingRequest = _faireRequests.ContainsKey(heroIdx);
            if (hasPendingRequest)
            {
                var req = _faireRequests[heroIdx];
                bool canAccept = gold >= req.amount;
                if (canAccept) result.faireRespond |= (byte)(1 << 1);
                result.faireRespond |= (byte)(1 << 2); // can always deny
            }

            return result;
        }

        /// <summary>Write action mask from packed ActionMaskBits (16 bytes, zero-allocation)</summary>
        private static void WriteActionMaskBinary(BinaryWriter w, ActionMaskBits mask)
        {
            w.Write(mask.skill);          // 1 byte
            w.Write(mask.unitTarget);     // 2 bytes
            w.Write(mask.skillLevelup);   // 1 byte
            w.Write(mask.statUpgrade);    // 2 bytes
            w.Write(mask.attribute);      // 1 byte
            w.Write(mask.itemBuy);        // 4 bytes (uint32)
            w.Write(mask.itemUse);        // 1 byte
            w.Write(mask.sealUse);        // 1 byte
            w.Write(mask.faireSend);      // 1 byte
            w.Write(mask.faireRequest);   // 1 byte
            w.Write(mask.faireRespond);   // 1 byte
        }

        // ============================================================
        // Binary State Builder (matches protocol.h)
        // ============================================================

        /// <summary>
        /// Convert FourCC string (e.g. "I00E") to a 16-bit index via simple hash.
        /// Returns 0 for null/empty.
        /// </summary>
        private static short FourCCToIndex16(string s)
        {
            if (string.IsNullOrEmpty(s) || s == "0000") return 0;
            // Use bottom 15 bits of FourCC int, set bit 15 to mark "valid"
            int fourcc = FourCC(s);
            return (short)((fourcc & 0x7FFF) | 0x4000);
        }

        /// <summary>
        /// Build binary state packet matching protocol.h StatePacket layout.
        /// Called from OnTick (JASS thread). Reuses same game logic as BuildStateJson.
        /// Also updates episodeState based on team wipe / score detection.
        /// </summary>
        private static byte[] BuildStateBinary()
        {
            float gameTime = tickCount * TICK_INTERVAL;
            int team0Alive = 0;
            int team1Alive = 0;

            // Get time of day
            float timeOfDay = 0f;
            try { timeOfDay = Natives.GetFloatGameState(Natives.ConvertFGameState(2)); }
            catch { }
            bool isNight = timeOfDay >= 18.0f || timeOfDay < 6.0f;
            float nextPointTime = 600f - (gameTime % 600f);
            if (nextPointTime > 599.9f) nextPointTime = 600f;

            // Reuse static buffers (zero-alloc)
            Array.Clear(_curX, 0, MAX_PLAYERS);
            Array.Clear(_curY, 0, MAX_PLAYERS);
            float[] curX = _curX;
            float[] curY = _curY;

            // Visibility check: we need players for IsUnitVisible
            JassPlayer[] players = _playersCache;
            for (int i = 0; i < MAX_PLAYERS; i++)
            {
                try { players[i] = Natives.Player(i); } catch { }
            }

            // Reuse static MemoryStream (zero-alloc per tick)
            _stateMs.SetLength(0);
            _stateMs.Position = 0;
            var w = _stateWriter;
            {
                // ---- PacketHeader (8 bytes) ----
                w.Write((ushort)0xFA7E);       // magic
                w.Write((byte)1);              // version
                w.Write((byte)1);              // msg_type = MSG_STATE
                w.Write((uint)tickCount);      // tick

                // ---- GlobalState (28 bytes) ----
                w.Write(gameTime);                                          // game_time (float)
                w.Write(timeOfDay);                                         // time_of_day (float)
                w.Write(nextPointTime);                                     // next_point_time (float)
                w.Write(isNight ? (byte)1 : (byte)0);                      // is_night (uint8)
                w.Write((byte)0); w.Write((byte)0); w.Write((byte)0);      // _pad_global[3]
                w.Write((short)JassStateCache.teamScore[0]);                // score_team0 (int16)
                w.Write((short)JassStateCache.teamScore[1]);                // score_team1 (int16)
                w.Write((short)TARGET_SCORE);                               // target_score (int16)
                w.Write((short)0);                                              // c_rank_stock field (deprecated, now per-player faire tracking)
                w.Write(0f);                                                // _reserved (float)

                // ---- UnitState x 12 ----
                for (int i = 0; i < MAX_PLAYERS; i++)
                {
                    if (!heroRegistered[i])
                    {
                        WriteEmptyUnitBinary(w, i);
                        continue;
                    }

                    JassUnit u = heroes[i];
                    int typeId = heroTypeIds[i];  // use registered typeId (morph-proof)
                    string heroIdStr = TypeIdToString(typeId);

                    // Basic stats
                    float hp = 0f, maxHp = 0f, mp = 0f, maxMp = 0f;
                    float x = 0f, y = 0f;
                    bool alive = false;
                    int lv = 0;
                    float moveSpd = 0f;
                    int str = 0, agi = 0, intel = 0;
                    float facing = 0f;

                    try { hp = Natives.GetUnitState(u, JassUnitState.Life); } catch { }
                    try { maxHp = Natives.GetUnitState(u, JassUnitState.MaxLife); } catch { }
                    try { mp = Natives.GetUnitState(u, JassUnitState.Mana); } catch { }
                    try { maxMp = Natives.GetUnitState(u, JassUnitState.MaxMana); } catch { }
                    try { x = Natives.GetUnitX(u); } catch { }
                    try { y = Natives.GetUnitY(u); } catch { }
                    alive = hp > 0.405f;
                    try { lv = (int)Natives.GetHeroLevel(u); } catch { }
                    try { moveSpd = Natives.GetUnitMoveSpeed(u); } catch { }
                    try { str = (int)Natives.GetHeroStr(u, true); } catch { }
                    try { agi = (int)Natives.GetHeroAgi(u, true); } catch { }
                    try { intel = (int)Natives.GetHeroInt(u, true); } catch { }
                    try { facing = Natives.GetUnitFacing(u); } catch { }

                    curX[i] = x;
                    curY[i] = y;

                    // Velocity
                    float velX = 0f, velY = 0f;
                    if (_prevPosValid)
                    {
                        velX = (x - _prevX[i]) / TICK_INTERVAL;
                        velY = (y - _prevY[i]) / TICK_INTERVAL;
                    }

                    // XP, skill points, stat points
                    int xp = 0, skillPoints = 0, statPoints = 0;
                    try { xp = (int)Natives.GetHeroXP(u); } catch { }
                    try { skillPoints = (int)Natives.GetHeroSkillPoints(u); } catch { }
                    statPoints = GetStatPoints(i);

                    // Attack, defense, attack speed, range
                    float atk = 0f, def = 0f, atkRange = 128f, atkSpd = 1.7f;
                    int mainStat = 0;
                    HeroData hdata;
                    if (_heroDataTable.TryGetValue(typeId, out hdata))
                    {
                        mainStat = hdata.mainStat;
                        int mainStatVal = mainStat == 0 ? str : mainStat == 1 ? agi : intel;
                        atk = hdata.baseAtk + mainStatVal * hdata.atkPerStr;
                        def = hdata.baseDef;
                        atkRange = hdata.atkRange;
                        atkSpd = hdata.baseAtkSpd;
                    }

                    // Revive remain placeholder
                    float reviveRemain = alive ? 0f : 10f;

                    // Track alive counts
                    if (alive)
                    {
                        if (i < 6) team0Alive++;
                        else team1Alive++;
                    }

                    // Buffs detection
                    bool isStunned = false, isSlowed = false, isSilenced = false;
                    bool isKnockback = false, isRooted = false, isInvuln = false;
                    try { isStunned = (int)Natives.GetUnitAbilityLevel(u, (JassObjectId)_buffStun) > 0; } catch { }
                    try { isSlowed = (int)Natives.GetUnitAbilityLevel(u, (JassObjectId)_buffSlow) > 0; } catch { }
                    try { isSilenced = (int)Natives.GetUnitAbilityLevel(u, (JassObjectId)_buffSilence) > 0; } catch { }
                    try { isRooted = (int)Natives.GetUnitAbilityLevel(u, (JassObjectId)_buffRoot) > 0; } catch { }
                    try { isInvuln = (int)Natives.GetUnitAbilityLevel(u, (JassObjectId)_buffInvuln) > 0; } catch { }

                    // Seal data
                    var sealData = GetSealItem(i);
                    int sealCharges = sealData.charges;

                    // Gold
                    int gold = GetPlayerGold(i);

                    // Visibility: per-player bitmask (bit j = player j can see this unit)
                    ushort visibleMask = 0;
                    for (int p = 0; p < MAX_PLAYERS; p++)
                    {
                        if (p == i) continue; // skip self
                        bool canSee = true;
                        if (_fogInitialized)
                        {
                            canSee = IsUnitVisibleViaFog(x, y, p);
                        }
                        else
                        {
                            try
                            {
                                if (players[p].Handle != IntPtr.Zero)
                                    canSee = Natives.IsUnitVisible(u, players[p]);
                            }
                            catch { canSee = false; }
                        }
                        if (canSee) visibleMask |= (ushort)(1 << p);
                    }

                    // Build action mask (zero-alloc packed bits)
                    var actionMask = BuildActionMaskBits(i, u, typeId, lv, alive, mp, gold, skillPoints, statPoints, gameTime);

                    // Debug: log mask values for p0 every 100 ticks
                    if (i == 0 && tickCount % 100 == 1)
                        Log($"[RLComm] DEBUG Mask p0: skillPts={skillPoints} statPts={statPoints} skillLvl=0x{actionMask.skillLevelup:X2} statUpg=0x{actionMask.statUpgrade:X4} attr=0x{actionMask.attribute:X2}");

                    // ---- Write UnitState ----
                    // Identity (6 bytes)
                    w.Write((byte)i);                                       // idx
                    byte[] heroIdBytes = TypeIdToBytes(typeId);
                    if (heroIdBytes.Length >= 4)
                        w.Write(heroIdBytes, 0, 4);                         // hero_id[4]
                    else
                    {
                        w.Write(heroIdBytes);
                        for (int pad = heroIdBytes.Length; pad < 4; pad++) w.Write((byte)0);
                    }
                    w.Write((byte)(i < 6 ? 0 : 1));                        // team

                    // Basic (37 bytes)
                    w.Write(hp);                                            // hp (float)
                    w.Write(maxHp);                                         // max_hp (float)
                    w.Write(mp);                                            // mp (float)
                    w.Write(maxMp);                                         // max_mp (float)
                    w.Write(x);                                             // x (float)
                    w.Write(y);                                             // y (float)
                    w.Write(velX);                                          // vel_x (float)
                    w.Write(velY);                                          // vel_y (float)
                    w.Write(alive ? (byte)1 : (byte)0);                     // alive (uint8)
                    w.Write(reviveRemain);                                  // revive_remain (float)

                    // Stats (24 bytes)
                    w.Write((short)str);                                    // str (int16)
                    w.Write((short)agi);                                    // agi (int16)
                    w.Write((short)intel);                                  // int_ (int16)
                    w.Write(atk);                                           // atk (float)
                    w.Write(def);                                           // def_ (float)
                    w.Write(moveSpd);                                       // move_spd (float)
                    w.Write(atkRange);                                      // atk_range (float)
                    w.Write(atkSpd);                                        // atk_spd (float)

                    // Progression (8 bytes)
                    w.Write((byte)lv);                                      // level (uint8)
                    w.Write((byte)skillPoints);                             // skill_points (uint8)
                    w.Write((byte)statPoints);                              // stat_points (uint8)
                    w.Write((byte)0);                                       // _pad_prog (uint8)
                    w.Write((int)xp);                                       // xp (int32)

                    // Skills: 6 slots (84 bytes = 6 * 14)
                    WriteSkillSlotsBinary(w, i, u, typeId, lv, gameTime);

                    // Upgrades (9 bytes)
                    for (int ui = 0; ui < 9; ui++)
                        w.Write((byte)JassStateCache.upgrades[i, ui]);

                    // Attributes (1 byte, bit-packed)
                    int attrCount = JassStateCache.attrCount[i];
                    byte attrBits = 0;
                    if (attrCount >= 1) attrBits |= 0x01;
                    if (attrCount >= 2) attrBits |= 0x02;
                    if (attrCount >= 3) attrBits |= 0x04;
                    if (attrCount >= 4) attrBits |= 0x08;
                    w.Write(attrBits);

                    // Buffs (1 byte, bit-packed: stun|slow|silence|knockback|root|invuln)
                    byte buffBits = 0;
                    if (isStunned)  buffBits |= 0x01;
                    if (isSlowed)   buffBits |= 0x02;
                    if (isSilenced) buffBits |= 0x04;
                    if (isKnockback) buffBits |= 0x08;
                    if (isRooted)   buffBits |= 0x10;
                    if (isInvuln)   buffBits |= 0x20;
                    w.Write(buffBits);

                    // Seal (8 bytes)
                    w.Write((byte)sealCharges);                             // seal_charges (uint8)
                    w.Write((short)JassStateCache.sealCd[i]);               // seal_cd (int16)
                    w.Write(JassStateCache.firstActive[i] != 0 ? (byte)1 : (byte)0); // seal_first_active (uint8)
                    w.Write((float)JassStateCache.firstRemain[i]);          // seal_first_remain (float)

                    // Items: 6 slots (24 bytes = 6 * 4)
                    WriteItemSlotsBinary(w, u);

                    // Economy (8 bytes)
                    w.Write((int)gold);                                     // faire (int32)
                    w.Write((short)FAIRE_CAP);                              // faire_cap (int16)
                    w.Write((byte)0); w.Write((byte)0);                     // _pad_econ[2]

                    // Flags (3 bytes)
                    w.Write(_alarmState[i] ? (byte)1 : (byte)0);           // enemy_alarm (uint8)
                    w.Write(visibleMask);                                   // visible_mask (uint16)

                    // Action Masks (16 bytes, bit-packed)
                    WriteActionMaskBinary(w, actionMask);
                }

                // Update previous positions
                for (int i = 0; i < MAX_PLAYERS; i++)
                {
                    _prevX[i] = curX[i];
                    _prevY[i] = curY[i];
                }
                _prevPosValid = true;

                // Episode state: score-based only (no team wipe — heroes respawn)
                if (episodeState == 0)
                {
                    if (JassStateCache.teamScore[0] >= TARGET_SCORE)
                        episodeState = 4;
                    else if (JassStateCache.teamScore[1] >= TARGET_SCORE)
                        episodeState = 4;
                }

                // ---- Events ----
                var events = _eventQueue.FlushAndClearBinary();
                int numEvents = Math.Min(events.Count, 32);
                w.Write((byte)numEvents);
                for (int e = 0; e < numEvents; e++)
                {
                    var evt = events[e];
                    w.Write(evt.type);
                    w.Write(evt.killerIdx);
                    w.Write(evt.victimIdx);
                    w.Write(evt.padding);
                    w.Write(evt.tick);
                }

                // ---- Pathability ----
                if (!_pathabilityComputed)
                {
                    try
                    {
                        _pathabilityGrid = ComputePathabilityGrid();
                        _pathabilityComputed = true;
                        Log("[RLComm] Pathability grid computed");
                    }
                    catch (Exception ex)
                    {
                        Log($"[RLComm] Pathability grid error: {ex.Message}");
                        _pathabilityGrid = new int[GRID_W * GRID_H];
                        _pathabilityComputed = true;
                    }
                }

                bool sendPath = (tickCount == 1 && _pathabilityGrid != null);
                w.Write(sendPath ? (byte)1 : (byte)0);                     // has_pathability
                if (sendPath)
                {
                    for (int c = 0; c < GRID_W * GRID_H; c++)
                        w.Write((byte)_pathabilityGrid[c]);
                }

                // ---- Visibility grids (zero-alloc, uses static buffers) ----
                try
                {
                    // Try fog bitmap direct read first (zero native calls, perfectly accurate)
                    if (_fogInitialized || InitFogPointers())
                    {
                        // Validate at tick 1, 10, 50, 100 to check fog bitmap convergence
                        if (tickCount == 1 || tickCount == 10 || tickCount == 50 || tickCount == 100)
                            ValidateFogMapping();

                        ComputeVisibilityFromFogBitmap(0, _visGridTeam0);
                        ComputeVisibilityFromFogBitmap(6, _visGridTeam1);
                    }
                    else
                    {
                        // Fallback: math-based circular vision
                        ComputeVisibilityGridInPlace(0, _visGridTeam0);
                        ComputeVisibilityGridInPlace(6, _visGridTeam1);
                    }
                    for (int c = 0; c < GRID_W * GRID_H; c++)
                        w.Write((byte)_visGridTeam0[c]);
                    for (int c = 0; c < GRID_W * GRID_H; c++)
                        w.Write((byte)_visGridTeam1[c]);
                }
                catch (Exception ex)
                {
                    Log($"[RLComm] Visibility grid error: {ex.Message}");
                    // Write zeros
                    for (int c = 0; c < GRID_W * GRID_H * 2; c++)
                        w.Write((byte)0);
                }

                // ---- Creep data ----
                try
                {
                    if (_creepGroup.Handle == IntPtr.Zero)
                        _creepGroup = Natives.CreateGroup();

                    // Enumerate neutral passive player's units
                    JassPlayer neutralPlayer = Natives.Player(15); // PLAYER_NEUTRAL_PASSIVE = 15
                    Natives.GroupEnumUnitsOfPlayer(_creepGroup, neutralPlayer, default(JassBooleanExpression));

                    // Collect creep data into temp arrays
                    int creepCount = 0;
                    float[] cxArr = new float[MAX_CREEPS];
                    float[] cyArr = new float[MAX_CREEPS];
                    float[] chpArr = new float[MAX_CREEPS];
                    float[] cmhpArr = new float[MAX_CREEPS];

                    JassUnit cu;
                    while ((cu = Natives.FirstOfGroup(_creepGroup)).Handle != IntPtr.Zero)
                    {
                        Natives.GroupRemoveUnit(_creepGroup, cu);
                        if (creepCount >= MAX_CREEPS) continue;

                        int cTypeId = 0;
                        try { cTypeId = (int)Natives.GetUnitTypeId(cu); } catch { continue; }

                        // Only track creep types h043, h044, h045
                        if (cTypeId != CREEP_TYPE_H043 && cTypeId != CREEP_TYPE_H044 && cTypeId != CREEP_TYPE_H045)
                            continue;

                        float cHp = 0f, cMaxHp = 0f, cX = 0f, cY = 0f;
                        try { cHp = Natives.GetUnitState(cu, JassUnitState.Life); } catch { }
                        try { cMaxHp = Natives.GetUnitState(cu, JassUnitState.MaxLife); } catch { }
                        try { cX = Natives.GetUnitX(cu); } catch { }
                        try { cY = Natives.GetUnitY(cu); } catch { }

                        if (cMaxHp <= 0f) continue;  // invalid

                        cxArr[creepCount] = cX;
                        cyArr[creepCount] = cY;
                        chpArr[creepCount] = cHp;
                        cmhpArr[creepCount] = cMaxHp;
                        creepCount++;
                    }

                    // Write num_creeps + CreepState array
                    w.Write((byte)creepCount);
                    for (int c = 0; c < creepCount; c++)
                    {
                        w.Write(cxArr[c]);    // float x
                        w.Write(cyArr[c]);    // float y
                        w.Write(chpArr[c]);   // float hp
                        w.Write(cmhpArr[c]);  // float max_hp
                    }
                }
                catch (Exception ex)
                {
                    Log($"[RLComm] Creep enum error: {ex.Message}");
                    w.Write((byte)0);  // num_creeps = 0
                }

                w.Flush();
                int len = (int)_stateMs.Position;
                if (_stateBuffer.Length < len)
                    _stateBuffer = new byte[len * 2];
                _stateMs.Position = 0;
                _stateMs.Read(_stateBuffer, 0, len);
                byte[] result = new byte[len];
                Buffer.BlockCopy(_stateBuffer, 0, result, 0, len);
                return result;
            }
        }

        /// <summary>Write an empty UnitState for unregistered hero slot</summary>
        private static void WriteEmptyUnitBinary(BinaryWriter w, int idx)
        {
            // Identity (6 bytes)
            w.Write((byte)idx);
            w.Write((byte)'0'); w.Write((byte)'0'); w.Write((byte)'0'); w.Write((byte)'0');
            w.Write((byte)(idx < 6 ? 0 : 1));

            // Basic (37 bytes) - all zeros
            w.Write(0f); w.Write(0f); w.Write(0f); w.Write(0f);  // hp, max_hp, mp, max_mp
            w.Write(0f); w.Write(0f);                              // x, y
            w.Write(0f); w.Write(0f);                              // vel_x, vel_y
            w.Write((byte)0);                                      // alive
            w.Write(0f);                                           // revive_remain

            // Stats (24 bytes) - all zeros
            w.Write((short)0); w.Write((short)0); w.Write((short)0);  // str, agi, int
            w.Write(0f); w.Write(0f); w.Write(0f); w.Write(0f); w.Write(0f); // atk, def, move_spd, atk_range, atk_spd

            // Progression (8 bytes)
            w.Write((byte)0); w.Write((byte)0); w.Write((byte)0); w.Write((byte)0); // level, skill_pts, stat_pts, pad
            w.Write((int)0);  // xp

            // Skills: 6 empty slots (84 bytes)
            for (int s = 0; s < 6; s++)
            {
                w.Write((int)0);   // abil_id
                w.Write((byte)0);  // level
                w.Write(0f);       // cd_remain
                w.Write(0f);       // cd_max
                w.Write((byte)0);  // exists
            }

            // Upgrades (9 bytes)
            for (int u = 0; u < 9; u++) w.Write((byte)0);

            // Attributes (1 byte)
            w.Write((byte)0);

            // Buffs (1 byte)
            w.Write((byte)0);

            // Seal (8 bytes)
            w.Write((byte)0); w.Write((short)0); w.Write((byte)0); w.Write(0f);

            // Items: 6 empty slots (24 bytes)
            for (int s = 0; s < 6; s++)
            {
                w.Write((short)0); w.Write((byte)0); w.Write((byte)0);
            }

            // Economy (8 bytes)
            w.Write((int)0); w.Write((short)FAIRE_CAP); w.Write((byte)0); w.Write((byte)0);

            // Flags (3 bytes)
            w.Write((byte)0); w.Write((ushort)0);

            // Action Masks (13 bytes) - empty mask
            // skill: bit 0 = none=true
            w.Write((byte)0x01);
            // unit_target: bits 12,13 = true
            w.Write((ushort)0x3000);
            // skill_levelup: bit 0 = true
            w.Write((byte)0x01);
            // stat_upgrade: bit 0 = true
            w.Write((ushort)0x0001);
            // attribute: bit 0 = true
            w.Write((byte)0x01);
            // item_buy: bit 0 = true (uint32)
            w.Write((uint)0x00000001);
            // item_use: bit 0 = true
            w.Write((byte)0x01);
            // seal_use: bit 0 = true
            w.Write((byte)0x01);
            // faire_send: bit 0 = true
            w.Write((byte)0x01);
            // faire_request: bit 0 = true
            w.Write((byte)0x01);
            // faire_respond: bit 0 = true
            w.Write((byte)0x01);
        }

        /// <summary>Write 6 SkillSlot entries (84 bytes total) for a unit</summary>
        private static void WriteSkillSlotsBinary(BinaryWriter w, int heroIdx, JassUnit u, int typeId, int heroLevel, float gameTime)
        {
            HeroData hdata;
            bool hasData = _heroDataTable.TryGetValue(typeId, out hdata);

            for (int s = 0; s < 6; s++)
            {
                int abilLevel = 0;
                float cdRemain = 0f;
                float cdMax = 0f;
                int abilId = 0;
                bool exists = false;

                if (hasData && s < hdata.skills.Length && hdata.skills[s].abilId != 0)
                {
                    SkillInfo si = hdata.skills[s];
                    abilId = si.abilId;
                    exists = true;
                    try { abilLevel = (int)Natives.GetUnitAbilityLevel(u, (JassObjectId)si.abilId); } catch { }
                    cdRemain = _cdTracker.GetCdRemain(heroIdx, s, gameTime);
                    if (abilLevel > 0 && abilLevel <= si.maxCd.Length)
                        cdMax = si.maxCd[abilLevel - 1];
                }

                w.Write((int)abilId);                                   // abil_id (int32)
                w.Write((byte)abilLevel);                               // level (uint8)
                w.Write(cdRemain);                                      // cd_remain (float)
                w.Write(cdMax);                                         // cd_max (float)
                w.Write(exists ? (byte)1 : (byte)0);                    // exists (uint8)
            }
        }

        /// <summary>Write 6 ItemSlot entries (24 bytes total) for a unit</summary>
        private static void WriteItemSlotsBinary(BinaryWriter w, JassUnit u)
        {
            for (int slot = 0; slot < 6; slot++)
            {
                short typeIdx = 0;
                byte charges = 0;

                try
                {
                    JassItem itm = Natives.UnitItemInSlot(u, slot);
                    if (itm.Handle != IntPtr.Zero)
                    {
                        int itmType = (int)Natives.GetItemTypeId(itm);
                        if (itmType != 0)
                        {
                            typeIdx = FourCCToIndex16(TypeIdToString(itmType));
                            charges = (byte)Math.Min(255, (int)Natives.GetItemCharges(itm));
                        }
                    }
                }
                catch { }

                w.Write(typeIdx);       // type_id (int16)
                w.Write(charges);       // charges (uint8)
                w.Write((byte)0);       // padding (uint8)
            }
        }

        /// <summary>Write action masks as bit-packed bytes (16 bytes total) -- LEGACY JObject version, kept for JSON debug path compatibility</summary>
        private static void WriteActionMaskBinary(BinaryWriter w, JObject actionMask)
        {
            // mask_skill (8 bits)
            byte maskSkill = 0;
            JArray skillArr = (JArray)actionMask["skill"];
            for (int b = 0; b < 8 && b < skillArr.Count; b++)
                if (skillArr[b].Value<bool>()) maskSkill |= (byte)(1 << b);
            w.Write(maskSkill);

            // mask_unit_target (16 bits)
            ushort maskUnitTarget = 0;
            JArray utArr = (JArray)actionMask["unit_target"];
            for (int b = 0; b < 16 && b < utArr.Count; b++)
                if (utArr[b].Value<bool>()) maskUnitTarget |= (ushort)(1 << b);
            w.Write(maskUnitTarget);

            // mask_skill_levelup (8 bits)
            byte maskSkillLvl = 0;
            JArray slArr = (JArray)actionMask["skill_levelup"];
            for (int b = 0; b < 8 && b < slArr.Count; b++)
                if (slArr[b].Value<bool>()) maskSkillLvl |= (byte)(1 << b);
            w.Write(maskSkillLvl);

            // mask_stat_upgrade (16 bits)
            ushort maskStatUpg = 0;
            JArray suArr = (JArray)actionMask["stat_upgrade"];
            for (int b = 0; b < 16 && b < suArr.Count; b++)
                if (suArr[b].Value<bool>()) maskStatUpg |= (ushort)(1 << b);
            w.Write(maskStatUpg);

            // mask_attribute (8 bits)
            byte maskAttr = 0;
            JArray atArr = (JArray)actionMask["attribute"];
            for (int b = 0; b < 8 && b < atArr.Count; b++)
                if (atArr[b].Value<bool>()) maskAttr |= (byte)(1 << b);
            w.Write(maskAttr);

            // mask_item_buy (32 bits → 17 used)
            uint maskItemBuy = 0;
            JArray ibArr = (JArray)actionMask["item_buy"];
            for (int b = 0; b < 32 && b < ibArr.Count; b++)
                if (ibArr[b].Value<bool>()) maskItemBuy |= (uint)(1 << b);
            w.Write(maskItemBuy);

            // mask_item_use (8 bits)
            byte maskItemUse = 0;
            JArray iuArr = (JArray)actionMask["item_use"];
            for (int b = 0; b < 8 && b < iuArr.Count; b++)
                if (iuArr[b].Value<bool>()) maskItemUse |= (byte)(1 << b);
            w.Write(maskItemUse);

            // mask_seal_use (8 bits)
            byte maskSealUse = 0;
            JArray seArr = (JArray)actionMask["seal_use"];
            for (int b = 0; b < 8 && b < seArr.Count; b++)
                if (seArr[b].Value<bool>()) maskSealUse |= (byte)(1 << b);
            w.Write(maskSealUse);

            // mask_faire_send (8 bits)
            byte maskFaireSend = 0;
            JArray fsArr = (JArray)actionMask["faire_send"];
            for (int b = 0; b < 8 && b < fsArr.Count; b++)
                if (fsArr[b].Value<bool>()) maskFaireSend |= (byte)(1 << b);
            w.Write(maskFaireSend);

            // mask_faire_request (8 bits)
            byte maskFaireReq = 0;
            JArray frArr = (JArray)actionMask["faire_request"];
            for (int b = 0; b < 8 && b < frArr.Count; b++)
                if (frArr[b].Value<bool>()) maskFaireReq |= (byte)(1 << b);
            w.Write(maskFaireReq);

            // mask_faire_respond (8 bits)
            byte maskFaireResp = 0;
            JArray frespArr = (JArray)actionMask["faire_respond"];
            for (int b = 0; b < 8 && b < frespArr.Count; b++)
                if (frespArr[b].Value<bool>()) maskFaireResp |= (byte)(1 << b);
            w.Write(maskFaireResp);
        }

        // ============================================================
        // Binary Done Packet Builder
        // ============================================================

        /// <summary>
        /// Build binary DONE packet (16 bytes) matching protocol.h DonePacket.
        /// </summary>
        private static byte[] BuildDoneBinary()
        {
            using (var ms = new MemoryStream(16))
            using (var w = new BinaryWriter(ms))
            {
                // PacketHeader (8 bytes)
                w.Write((ushort)0xFA7E);       // magic
                w.Write((byte)1);              // version
                w.Write((byte)3);              // msg_type = MSG_DONE
                w.Write((uint)tickCount);      // tick

                // Determine winner and reason
                byte winner = 2; // draw
                byte reason = 0;
                if (episodeState == 1) { winner = 1; reason = 1; }       // team0 wiped = team1 wins, team_wipe
                else if (episodeState == 2) { winner = 0; reason = 1; }  // team1 wiped = team0 wins, team_wipe
                else if (episodeState == 3)
                {
                    reason = 2; // timeout
                    if (JassStateCache.teamScore[0] > JassStateCache.teamScore[1]) winner = 0;
                    else if (JassStateCache.teamScore[1] > JassStateCache.teamScore[0]) winner = 1;
                    else winner = 2;
                }
                else if (episodeState == 4)
                {
                    reason = 3; // score
                    winner = JassStateCache.teamScore[0] >= TARGET_SCORE ? (byte)0 : (byte)1;
                }

                w.Write(winner);                                        // winner (uint8)
                w.Write(reason);                                        // reason (uint8)
                w.Write((short)JassStateCache.teamScore[0]);            // score_team0 (int16)
                w.Write((short)JassStateCache.teamScore[1]);            // score_team1 (int16)
                w.Write((byte)0); w.Write((byte)0);                     // _pad[2]

                w.Flush();
                return ms.ToArray();
            }
        }

        // ============================================================
        // Binary Action Processor
        // ============================================================

        /// <summary>
        /// Process binary ActionPacket (344 bytes) matching protocol.h.
        /// </summary>
        private static void ProcessActionBinary(byte[] data)
        {
            try
            {
                using (var ms = new MemoryStream(data))
                using (var r = new BinaryReader(ms))
                {
                    // Read header (8 bytes)
                    ushort magic = r.ReadUInt16();
                    if (magic != 0xFA7E) return;
                    byte version = r.ReadByte();
                    byte msgType = r.ReadByte();
                    if (msgType != 2) return; // MSG_ACTION
                    uint tick = r.ReadUInt32();

                    // Read 12 UnitAction (28 bytes each)
                    for (int i = 0; i < MAX_PLAYERS; i++)
                    {
                        byte idx = r.ReadByte();
                        byte pad = r.ReadByte();
                        float moveX = r.ReadSingle();
                        float moveY = r.ReadSingle();
                        float pointX = r.ReadSingle();
                        float pointY = r.ReadSingle();
                        byte skill = r.ReadByte();
                        byte unitTarget = r.ReadByte();
                        byte skillLevelup = r.ReadByte();
                        byte statUpgrade = r.ReadByte();
                        byte attribute = r.ReadByte();
                        byte itemBuy = r.ReadByte();
                        byte itemUse = r.ReadByte();
                        byte sealUse = r.ReadByte();
                        byte faireSend = r.ReadByte();
                        byte faireRequest = r.ReadByte();
                        byte faireRespond = r.ReadByte();
                        byte pad2 = r.ReadByte();

                        // Validate and execute
                        if (idx >= MAX_PLAYERS || !heroRegistered[idx]) continue;
                        if (episodeState != 0) continue;

                        JassUnit u = heroes[idx];
                        float gameTime = tickCount * TICK_INTERVAL;

                        // Check alive (except seal_use=5 revive)
                        bool alive = IsUnitAlive(u);

                        // 9. Seal use (process first since revive can bring dead hero back)
                        if (sealUse > 0)
                            ExecuteSealUse(idx, u, sealUse, pointX, pointY);

                        // Re-check alive after potential revive
                        alive = IsUnitAlive(u);
                        if (!alive) continue;

                        float cx = 0f, cy = 0f;
                        try { cx = Natives.GetUnitX(u); } catch { }
                        try { cy = Natives.GetUnitY(u); } catch { }

                        // 4. Skill levelup
                        if (skillLevelup > 0)
                            ExecuteSkillLevelup(idx, u, skillLevelup);

                        // 5. Stat upgrade
                        if (statUpgrade > 0)
                            ExecuteStatUpgrade(idx, u, statUpgrade);

                        // 6. Attribute
                        if (attribute > 0)
                            ExecuteAttribute(idx, u, attribute);

                        // 7. Item buy
                        if (itemBuy > 0)
                            ExecuteItemBuy(idx, u, itemBuy);

                        // 8. Item use
                        if (itemUse > 0)
                            ExecuteItemUse(idx, u, itemUse, unitTarget, pointX, pointY, cx, cy);

                        // 10. Faire send
                        if (faireSend > 0)
                            ExecuteFaireSend(idx, faireSend);

                        // 11. Faire request
                        if (faireRequest > 0)
                            ExecuteFaireRequest(idx, faireRequest);

                        // 12. Faire respond
                        if (faireRespond > 0)
                            ExecuteFaireRespond(idx, faireRespond);

                        // 3. Skill (takes priority over move if skill is actually used)
                        bool skillIssued = false;
                        if (skill >= 1)
                        {
                            skillIssued = ExecuteSkill(idx, u, skill, unitTarget, pointX, pointY, cx, cy, gameTime);
                        }

                        // 1. Move (only if no skill was issued AND move has meaningful direction)
                        //    Skip if move is near-zero to avoid cancelling ongoing attacks/skills
                        //    Also skip during targeting skill grace period to prevent cancellation
                        if (!skillIssued)
                        {
                            bool inGracePeriod = (tickCount - _targetSkillGraceTick[idx]) < TARGETING_SKILL_GRACE
                                                 && _targetSkillGraceTick[idx] > 0;
                            if (inGracePeriod)
                            {
                                // Targeting skill in progress, suppress move to let it complete
                            }
                            else
                            {
                                float moveNorm = (float)Math.Sqrt(moveX * moveX + moveY * moveY);
                                if (moveNorm >= 0.1f)
                                {
                                    ExecuteMove(idx, u, moveX, moveY, cx, cy);
                                }
                                // else: no-op, let unit continue current action
                            }
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Log($"[RLComm] ProcessActionBinary error: {ex.Message}");
            }
        }

        // ============================================================
        // JSON Action Processor (12 action heads) -- kept for backward compat
        // ============================================================

        /// <summary>
        /// Process JSON actions from Python.
        /// Format: {"actions": [{"idx":0, "move_x":0.35, "move_y":0.72, "skill":2, ...}, ...]}
        /// </summary>
        private static void ProcessActionJson(string json)
        {
            if (string.IsNullOrEmpty(json)) return;

            try
            {
                JObject root = JObject.Parse(json);
                JArray actions = (JArray)root["actions"];

                foreach (JToken token in actions)
                {
                    int idx = token.Value<int>("idx");
                    if (idx < 0 || idx >= MAX_PLAYERS || !heroRegistered[idx]) continue;
                    if (episodeState != 0) continue;

                    JassUnit u = heroes[idx];
                    float gameTime = tickCount * TICK_INTERVAL;

                    // Check alive (except seal_use=5 revive)
                    bool alive = IsUnitAlive(u);

                    // Read all action heads
                    float moveX = token.Value<float>("move_x");
                    float moveY = token.Value<float>("move_y");
                    int skill = token.Value<int>("skill");
                    int unitTarget = token.Value<int>("unit_target");
                    float pointX = token.Value<float>("point_x");
                    float pointY = token.Value<float>("point_y");
                    int skillLevelup = token.Value<int>("skill_levelup");
                    int statUpgrade = token.Value<int>("stat_upgrade");
                    int attribute = token.Value<int>("attribute");
                    int itemBuy = token.Value<int>("item_buy");
                    int itemUse = token.Value<int>("item_use");
                    int sealUse = token.Value<int>("seal_use");
                    int faireSend = token.Value<int>("faire_send");
                    int faireRequest = token.Value<int>("faire_request");
                    int faireRespond = token.Value<int>("faire_respond");

                    // 9. Seal use (process first since revive can bring dead hero back)
                    if (sealUse > 0)
                        ExecuteSealUse(idx, u, sealUse, pointX, pointY);

                    // Re-check alive after potential revive
                    alive = IsUnitAlive(u);
                    if (!alive) continue;

                    float cx = 0f, cy = 0f;
                    try { cx = Natives.GetUnitX(u); } catch { }
                    try { cy = Natives.GetUnitY(u); } catch { }

                    // 4. Skill levelup (can happen alongside other actions)
                    if (skillLevelup > 0)
                        ExecuteSkillLevelup(idx, u, skillLevelup);

                    // 5. Stat upgrade
                    if (statUpgrade > 0)
                        ExecuteStatUpgrade(idx, u, statUpgrade);

                    // 6. Attribute
                    if (attribute > 0)
                        ExecuteAttribute(idx, u, attribute);

                    // 7. Item buy
                    if (itemBuy > 0)
                        ExecuteItemBuy(idx, u, itemBuy);

                    // 8. Item use
                    if (itemUse > 0)
                        ExecuteItemUse(idx, u, itemUse, unitTarget, pointX, pointY, cx, cy);

                    // 10. Faire send
                    if (faireSend > 0)
                        ExecuteFaireSend(idx, faireSend);

                    // 11. Faire request
                    if (faireRequest > 0)
                        ExecuteFaireRequest(idx, faireRequest);

                    // 12. Faire respond
                    if (faireRespond > 0)
                        ExecuteFaireRespond(idx, faireRespond);

                    // 3. Skill (takes priority over move if skill is actually used)
                    bool skillIssued = false;
                    if (skill >= 1)
                    {
                        skillIssued = ExecuteSkill(idx, u, skill, unitTarget, pointX, pointY, cx, cy, gameTime);
                    }

                    // 1. Move (only if no skill was issued)
                    if (!skillIssued)
                    {
                        ExecuteMove(idx, u, moveX, moveY, cx, cy);
                    }
                }
            }
            catch (Exception ex)
            {
                Log($"[RLComm] ProcessActionJson error: {ex.Message}");
            }
        }

        // ============================================================
        // Execute* Methods (12 action heads)
        // ============================================================

        /// <summary>
        /// Head 1: Execute continuous move: move_x, move_y in [-1,1].
        /// If norm less than 0.1, issue stop. Else normalize and move.
        /// </summary>
        private static void ExecuteMove(int idx, JassUnit u, float moveX, float moveY, float cx, float cy)
        {
            try
            {
                float norm = (float)Math.Sqrt(moveX * moveX + moveY * moveY);
                if (norm < 0.1f)
                {
                    // No-op: let unit continue current action (don't cancel attacks/skills)
                    return;
                }

                // Normalize
                float dirX = moveX / norm;
                float dirY = moveY / norm;

                float ms = Natives.GetUnitMoveSpeed(u);
                float dist = ms * 0.5f; // move about 0.5s worth
                float tx = cx + dirX * dist;
                float ty = cy + dirY * dist;

                // Clamp to map bounds
                tx = Clampf(tx, MAP_MIN_X + 64, MAP_MAX_X - 64);
                ty = Clampf(ty, MAP_MIN_Y + 64, MAP_MAX_Y - 64);

                Natives.IssuePointOrder(u, "move", tx, ty);
            }
            catch (Exception ex)
            {
                Log($"[RLComm] ExecuteMove error idx={idx}: {ex.Message}");
            }
        }

        // Skill execution statistics (logged every 500 ticks)
        private static int _skillStatImmediate;
        private static int _skillStatUnitTarget;
        private static int _skillStatPointTarget;
        private static int _skillStatAttack;
        private static int _skillStatFailed;
        private static int _skillStatLastLogTick;
        // Per-slot counts: [0]=none [1]=atk [2]=Q [3]=W [4]=E [5]=R [6]=D [7]=F
        private static int[] _skillSlotCounts = new int[8];
        // Per-hero skill use log (recent, up to 20 entries)
        private static string[] _recentSkillLog = new string[20];
        private static int _recentSkillIdx;

        /// <summary>
        /// Head 2 + 3a + 3b: Execute skill: 0=none, 1=attack, 2=Q, 3=W, 4=E, 5=R, 6=D, 7=F
        /// unit_target: 0-5=ally, 6-11=enemy, 12=self, 13=point_mode
        /// Returns true if a skill/attack order was issued.
        /// </summary>
        private static bool ExecuteSkill(int idx, JassUnit u, int skill, int unitTarget,
            float pointX, float pointY, float cx, float cy, float gameTime)
        {
            if (skill == 0) return false;

            // Periodic skill stats logging
            if (tickCount - _skillStatLastLogTick >= 500)
            {
                if (_skillStatLastLogTick > 0)
                {
                    Log($"[RLComm] SkillStats(last500): imm={_skillStatImmediate} utgt={_skillStatUnitTarget} ptgt={_skillStatPointTarget} atk={_skillStatAttack} fail={_skillStatFailed}");
                    Log($"[RLComm] SkillSlots(last500): Q={_skillSlotCounts[2]} W={_skillSlotCounts[3]} E={_skillSlotCounts[4]} R={_skillSlotCounts[5]} D={_skillSlotCounts[6]} F={_skillSlotCounts[7]}");
                    // Log recent skill uses
                    var sb = new System.Text.StringBuilder("[RLComm] RecentSkills: ");
                    for (int r = 0; r < 20; r++)
                    {
                        int ri = (_recentSkillIdx - 20 + r + 200) % 20;
                        if (_recentSkillLog[ri] != null) sb.Append(_recentSkillLog[ri]).Append(" | ");
                    }
                    Log(sb.ToString());
                }
                _skillStatImmediate = 0; _skillStatUnitTarget = 0; _skillStatPointTarget = 0;
                _skillStatAttack = 0; _skillStatFailed = 0;
                for (int s = 0; s < 8; s++) _skillSlotCounts[s] = 0;
                _skillStatLastLogTick = tickCount;
            }

            try
            {
                int typeId = (int)Natives.GetUnitTypeId(u);
                int myTeamBase = idx < 6 ? 0 : 6;
                int enemyTeamBase = idx < 6 ? 6 : 0;

                // 1 = Attack
                if (skill == 1)
                {
                    // Block friendly-fire: attack on ally unit targets (0-5)
                    if (unitTarget < 6)
                    {
                        _skillStatFailed++;
                        return false;
                    }

                    JassUnit target = ResolveTargetUnit(idx, unitTarget, myTeamBase, enemyTeamBase);
                    if (target.Handle != IntPtr.Zero && unitTarget <= 12)
                    {
                        Natives.IssueTargetOrder(u, "attack", target);
                        _skillStatAttack++;
                        return true;
                    }
                    else if (unitTarget == 13) // point mode
                    {
                        float tx, ty;
                        DecodePolarPoint(pointX, pointY, 600f, cx, cy, out tx, out ty);
                        Natives.IssuePointOrder(u, "attack", tx, ty);
                        _skillStatAttack++;
                        return true;
                    }
                    _skillStatFailed++;
                    return false;
                }

                // 2-7 = Q/W/E/R/D/F (mapped to slot 0-5)
                int slotIdx = skill - 2; // 0=Q,1=W,2=E,3=R,4=D,5=F
                if (slotIdx < 0 || slotIdx > 5) return false;

                HeroData hdata;
                if (!_heroDataTable.TryGetValue(typeId, out hdata)) return false;
                if (slotIdx >= hdata.skills.Length) return false;

                SkillInfo si = hdata.skills[slotIdx];
                if (si.abilId == 0) return false;

                int abilLevel = 0;
                try { abilLevel = (int)Natives.GetUnitAbilityLevel(u, (JassObjectId)si.abilId); } catch { }
                if (abilLevel <= 0) return false;

                string[] slotNames = { "Q", "W", "E", "R", "D", "F" };
                string slotName = slotIdx < slotNames.Length ? slotNames[slotIdx] : "?";
                _skillSlotCounts[skill] = _skillSlotCounts[skill] + 1;

                // Use string-based orders matching the ability's actual order string
                // For ANcl abilities: orderId = Ncl6 (Channel base order ID)
                // For non-ANcl: orderId = aord override or base type inherent order
                string orderStr = si.orderId;

                // Skip abilities with no order string (ANfd, AIh2, Afod base types)
                // These cannot be issued via IssueOrder - need map patch or JASS relay
                if (string.IsNullOrEmpty(orderStr))
                {
                    _skillStatFailed++;
                    _recentSkillLog[_recentSkillIdx % 20] = $"p{idx}{slotName}:NOORDER({si.abilId:X})";
                    _recentSkillIdx++;
                    return false;
                }

                bool orderOk = false;

                if (si.targetType == 0) // immediate (no target)
                {
                    bool ok = Natives.IssueImmediateOrder(u, orderStr);
                    if (ok)
                    {
                        _targetSkillGraceTick[idx] = tickCount; // prevent next-tick move from cancelling channel
                        _skillStatImmediate++;
                        _recentSkillLog[_recentSkillIdx % 20] = $"p{idx}{slotName}imm({orderStr}):OK";
                        orderOk = true;
                    }
                    else
                    {
                        _skillStatFailed++;
                        float mp2 = Natives.GetUnitState(u, JassUnitState.Mana);
                        _recentSkillLog[_recentSkillIdx % 20] = $"p{idx}{slotName}imm({orderStr}):FAIL[lv{abilLevel},mp{mp2:F0}]";
                    }
                    _recentSkillIdx++;
                }
                else if (si.targetType == 1) // unit target
                {
                    JassUnit target;
                    if (unitTarget == 13)
                    {
                        target = FindNearestEnemy(idx, cx, cy, myTeamBase, enemyTeamBase);
                    }
                    else
                    {
                        target = ResolveTargetUnit(idx, unitTarget, myTeamBase, enemyTeamBase);
                    }

                    if (target.Handle != IntPtr.Zero)
                    {
                        bool ok = Natives.IssueTargetOrder(u, orderStr, target);
                        _targetSkillGraceTick[idx] = tickCount;
                        if (ok)
                        {
                            _skillStatUnitTarget++;
                            _recentSkillLog[_recentSkillIdx % 20] = $"p{idx}{slotName}utgt({orderStr}):OK";
                            orderOk = true;
                        }
                        else
                        {
                            _skillStatFailed++;
                            _recentSkillLog[_recentSkillIdx % 20] = $"p{idx}{slotName}utgt({orderStr}):FAIL";
                        }
                        _recentSkillIdx++;
                    }
                    else
                    {
                        _skillStatFailed++;
                        _recentSkillLog[_recentSkillIdx % 20] = $"p{idx}{slotName}utgt:NOTGT";
                        _recentSkillIdx++;
                        return false;
                    }
                }
                else if (si.targetType == 2) // point target
                {
                    float tx, ty;
                    if (unitTarget >= 6 && unitTarget <= 11)
                    {
                        // Target an enemy hero's position (ally positions excluded — wasteful for point skills)
                        JassUnit target = ResolveTargetUnit(idx, unitTarget, myTeamBase, enemyTeamBase);
                        if (target.Handle != IntPtr.Zero)
                        {
                            tx = Natives.GetUnitX(target);
                            ty = Natives.GetUnitY(target);
                        }
                        else
                        {
                            DecodePolarPoint(pointX, pointY, si.maxRange, cx, cy, out tx, out ty);
                        }
                    }
                    else // unitTarget == 12 (self), 13 (point_mode), or default
                    {
                        DecodePolarPoint(pointX, pointY, si.maxRange, cx, cy, out tx, out ty);
                    }

                    bool ok = Natives.IssuePointOrder(u, orderStr, tx, ty);
                    _targetSkillGraceTick[idx] = tickCount;
                    if (ok)
                    {
                        _skillStatPointTarget++;
                        _recentSkillLog[_recentSkillIdx % 20] = $"p{idx}{slotName}pt({orderStr},{tx:F0},{ty:F0}):OK";
                        orderOk = true;
                    }
                    else
                    {
                        _skillStatFailed++;
                        float mp = Natives.GetUnitState(u, JassUnitState.Mana);
                        float hp = Natives.GetUnitState(u, JassUnitState.Life);
                        _recentSkillLog[_recentSkillIdx % 20] = $"p{idx}{slotName}pt({orderStr},{tx:F0},{ty:F0}):FAIL[lv{abilLevel},hp{hp:F0},mp{mp:F0}]";
                    }
                    _recentSkillIdx++;
                }

                // Track cooldown
                if (orderOk)
                {
                    float maxCd = 0f;
                    if (abilLevel > 0 && abilLevel <= si.maxCd.Length)
                        maxCd = si.maxCd[abilLevel - 1];
                    if (maxCd > 0f)
                    {
                        _cdTracker.OnSkillUsed(idx, slotIdx, gameTime, maxCd);
                        // Shared cooldown group: apply CD to all skills in same group
                        if (si.sharedCdGroup > 0)
                        {
                            for (int gs = 0; gs < hdata.skills.Length; gs++)
                            {
                                if (gs != slotIdx && hdata.skills[gs].sharedCdGroup == si.sharedCdGroup)
                                    _cdTracker.OnSkillUsed(idx, gs, gameTime, maxCd);
                            }
                        }
                    }
                }
                else
                {
                    // Prevent retry spam: block this skill for 5s after failure
                    _cdTracker.OnSkillUsed(idx, slotIdx, gameTime, 5.0f);
                }

                return true;
            }
            catch (Exception ex)
            {
                Log($"[RLComm] ExecuteSkill error idx={idx} skill={skill}: {ex.Message}");
                return false;
            }
        }

        /// <summary>Resolve unit_target index to a JassUnit</summary>
        private static JassUnit ResolveTargetUnit(int heroIdx, int unitTarget, int myTeamBase, int enemyTeamBase)
        {
            if (unitTarget >= 0 && unitTarget <= 5)
            {
                // Ally
                int pid = myTeamBase + unitTarget;
                if (heroRegistered[pid] && IsUnitAlive(heroes[pid]))
                    return heroes[pid];
            }
            else if (unitTarget >= 6 && unitTarget <= 11)
            {
                // Enemy
                int pid = enemyTeamBase + (unitTarget - 6);
                if (heroRegistered[pid] && IsUnitAlive(heroes[pid]))
                    return heroes[pid];
            }
            else if (unitTarget == 12)
            {
                // Self
                return heroes[heroIdx];
            }
            return default(JassUnit);
        }

        /// <summary>Find nearest alive enemy hero for unit-target skill fallback</summary>
        private static JassUnit FindNearestEnemy(int heroIdx, float cx, float cy, int myTeamBase, int enemyTeamBase)
        {
            float bestDist = float.MaxValue;
            JassUnit bestTarget = default(JassUnit);
            for (int e = 0; e < 6; e++)
            {
                int pid = enemyTeamBase + e;
                if (!heroRegistered[pid] || !IsUnitAlive(heroes[pid])) continue;
                float ex = 0f, ey = 0f;
                try { ex = Natives.GetUnitX(heroes[pid]); } catch { continue; }
                try { ey = Natives.GetUnitY(heroes[pid]); } catch { continue; }
                float dist = (ex - cx) * (ex - cx) + (ey - cy) * (ey - cy);
                if (dist < bestDist)
                {
                    bestDist = dist;
                    bestTarget = heroes[pid];
                }
            }
            return bestTarget;
        }

        /// <summary>
        /// Head 4: Execute skill levelup: 0=none, 1=Q, 2=W, 3=E, 4=R, 5=allstat(+2/+2/+2)
        /// </summary>
        private static void ExecuteSkillLevelup(int idx, JassUnit u, int skillLevelup)
        {
            if (skillLevelup == 0) return;

            try
            {
                int skillPoints = (int)Natives.GetHeroSkillPoints(u);
                if (skillPoints <= 0) return;

                if (skillLevelup >= 1 && skillLevelup <= 4)
                {
                    int typeId = (int)Natives.GetUnitTypeId(u);
                    HeroData hdata;
                    if (!_heroDataTable.TryGetValue(typeId, out hdata)) return;

                    int slotIdx = skillLevelup - 1; // 0=Q,1=W,2=E,3=R
                    if (slotIdx >= hdata.skills.Length || hdata.skills[slotIdx].abilId == 0) return;

                    int learnId = hdata.skills[slotIdx].learnAbilId != 0
                        ? hdata.skills[slotIdx].learnAbilId
                        : hdata.skills[slotIdx].abilId;
                    Natives.SelectHeroSkill(u, (JassObjectId)learnId);
                    Log($"[RLComm] SkillLevelup: p{idx} slot={slotIdx}");
                }
                else if (skillLevelup == 5)
                {
                    // All-stat: +2 str, +2 agi, +2 int (requires all QWER at level 5)
                    int typeId = (int)Natives.GetUnitTypeId(u);
                    HeroData hdata;
                    if (!_heroDataTable.TryGetValue(typeId, out hdata)) return;

                    // Verify all QWER at level 5
                    for (int s = 0; s < 4; s++)
                    {
                        if (s >= hdata.skills.Length || hdata.skills[s].abilId == 0) return;
                        int lv = (int)Natives.GetUnitAbilityLevel(u, (JassObjectId)hdata.skills[s].abilId);
                        if (lv < 5) return;
                    }

                    int curStr = (int)Natives.GetHeroStr(u, false);
                    int curAgi = (int)Natives.GetHeroAgi(u, false);
                    int curInt = (int)Natives.GetHeroInt(u, false);
                    Natives.SetHeroStr(u, curStr + 2, true);
                    Natives.SetHeroAgi(u, curAgi + 2, true);
                    Natives.SetHeroInt(u, curInt + 2, true);
                    // Deduct skill point
                    Natives.UnitModifySkillPoints(u, -1);
                    Log($"[RLComm] AllStat: p{idx} str+2 agi+2 int+2");
                }
            }
            catch (Exception ex)
            {
                Log($"[RLComm] ExecuteSkillLevelup error idx={idx}: {ex.Message}");
            }
        }

        /// <summary>
        /// Head 5: Execute stat upgrade via grail: 0=none, 1-9=upgrade types
        /// 1=str, 2=agi, 3=int, 4=HP regen, 5=MP regen, 6=faire regen,
        /// 7=attack, 8=defense, 9=movespeed
        /// </summary>
        // Grail ability order strings: index = statUpgrade (1-9)
        // Issuing these on the grail unit triggers JASS s__GrailSupport_onUpgrade,
        // which handles stat modification, s__User_upgradeList tracking, point deduction, and UI display.
        private static readonly string[] GRAIL_ORDER_STRINGS = {
            "",              // 0: unused
            "absorb",        // 1: STR+1      (A02W, AAns)
            "acidbomb",      // 2: AGI+1      (A03D, AAns)
            "ambush",        // 3: INT+1      (A03E, AAns)
            "slow",          // 4: HP regen   (A03Y, AAns)
            "blizzard",      // 5: MP regen   (A03Z, AAns)
            "unrobogoblin",  // 6: Gold regen (A0A9, AAns)
            "avatar",        // 7: Attack+3   (A03W, AAns)
            "animatedead",   // 8: Defense+2  (A03X, AAns)
            "firebolt",      // 9: MoveSpd+5  (A04Y, AAns)
        };

        private static void ExecuteStatUpgrade(int idx, JassUnit u, int statUpgrade)
        {
            if (statUpgrade == 0) return;
            if (statUpgrade < 1 || statUpgrade > 9) return;
            if (!_grailRegistered[idx]) return;

            try
            {
                int statPts = GetStatPoints(idx);
                if (statPts <= 0) return;

                // Check cap
                int upgradeIdx = statUpgrade - 1;
                if (JassStateCache.upgrades[idx, upgradeIdx] >= STAT_UPGRADE_CAPS[upgradeIdx])
                {
                    Log($"[RLComm] StatUpgrade: p{idx} upgrade={statUpgrade} at cap");
                    return;
                }

                // Issue order on the grail unit — triggers JASS GrailSupport_onUpgrade
                // which handles SetHeroStr/Agi/Int, SetPlayerTechResearched, s__User_upgrade,
                // s__User_modPoint (point deduction), and F3+C display tracking.
                JassUnit grail = _grailUnits[idx];
                string orderStr = GRAIL_ORDER_STRINGS[statUpgrade];
                bool ok = Natives.IssueImmediateOrder(grail, orderStr);

                if (ok)
                {
                    // Update local cache (JASS already handled the actual upgrade)
                    JassStateCache.upgrades[idx, upgradeIdx]++;
                    Log($"[RLComm] StatUpgrade: p{idx} type={statUpgrade} order={orderStr} newLevel={JassStateCache.upgrades[idx, upgradeIdx]}");
                }
                else
                {
                    Log($"[RLComm] StatUpgrade: p{idx} type={statUpgrade} order={orderStr} FAILED");
                }
            }
            catch (Exception ex)
            {
                Log($"[RLComm] ExecuteStatUpgrade error idx={idx}: {ex.Message}");
            }
        }

        /// <summary>
        /// Head 6: Execute attribute selection: 0=none, 1-4=A/B/C/D
        /// Attribute abilities live on the grail's spellbook. Issuing the order on the
        /// grail triggers JASS attributeFirst/Second/Third/Fourth handlers for the hero.
        /// JASS handles stat point deduction, attribute effects, and tracking.
        /// No sequential constraint — any attribute can be chosen freely.
        /// </summary>

        // Order strings per attribute slot (base types: Absk=berserk, Aroa=roar, Atau=taunt, ACwe=web)
        // All heroes share the same base type per slot, so the order string is uniform.
        // Exception: SaberAlter (H007) attr3 uses ANcl with order "nagabuild" instead of "taunt".
        private static readonly string[] ATTR_ORDER_STRINGS = {
            "",          // 0: unused
            "berserk",   // 1: attr1 (base=Absk)
            "roar",      // 2: attr2 (base=Aroa)
            "taunt",     // 3: attr3 (base=Atau)
            "web",       // 4: attr4 (base=ACwe)
        };

        private static void ExecuteAttribute(int idx, JassUnit u, int attribute)
        {
            if (attribute == 0) return;
            if (attribute < 1 || attribute > 4) return;

            try
            {
                // Check grail exists
                if (!_grailRegistered[idx])
                {
                    Log($"[RLComm] Attribute: p{idx} no grail registered");
                    return;
                }

                int typeId = (int)Natives.GetUnitTypeId(u);
                HeroData hdata;
                bool hasData = _heroDataTable.TryGetValue(typeId, out hdata);

                // Check if already acquired
                // Exception: Lancelot (H03M) attr4 can be acquired twice (JASS returns false on 1st use, keeping trigger alive)
                bool isLancelotAttr4 = (attribute == 4 && typeId == FourCC("H03M"));
                int acquireCount = JassStateCache.attrAcquired[idx, attribute];
                int maxAcquire = isLancelotAttr4 ? 2 : 1;
                if (acquireCount >= maxAcquire)
                {
                    Log($"[RLComm] Attribute: p{idx} attr={attribute} already acquired ({acquireCount}/{maxAcquire})");
                    return;
                }

                // Check stat points cost (pre-validation; JASS handles actual deduction)
                int cost = 10; // default
                if (hasData && hdata.attributeCost != null && attribute - 1 < hdata.attributeCost.Length)
                    cost = hdata.attributeCost[attribute - 1];

                int statPts = GetStatPoints(idx);
                if (statPts < cost)
                {
                    Log($"[RLComm] Attribute: p{idx} not enough stat points ({statPts} < {cost})");
                    return;
                }

                // Determine order string (SaberAlter attr3 exception)
                string orderStr = ATTR_ORDER_STRINGS[attribute];
                if (attribute == 3 && typeId == FourCC("H007"))
                    orderStr = "nagabuild"; // SaberAlter attr3 uses ANcl instead of Atau

                // Issue order on grail — attribute abilities are in the grail's spellbook.
                // Triggers JASS EVENT_UNIT_SPELL_EFFECT on grail → attributeFirst/Second/Third/Fourth
                // which applies effects, deducts stat points via s__User_modPoint, and removes the attribute unit.
                JassUnit grail = _grailUnits[idx];
                bool ok = Natives.IssueImmediateOrder(grail, orderStr);

                if (ok)
                {
                    JassStateCache.attrAcquired[idx, attribute]++;
                    JassStateCache.attrCount[idx]++;
                    Log($"[RLComm] Attribute: p{idx} attr={attribute} order={orderStr} ok=true (acquired={JassStateCache.attrAcquired[idx, attribute]}/{maxAcquire}, total={JassStateCache.attrCount[idx]})");
                }
                else
                {
                    Log($"[RLComm] Attribute: p{idx} attr={attribute} order={orderStr} ok=false");
                }
            }
            catch (Exception ex)
            {
                Log($"[RLComm] ExecuteAttribute error idx={idx}: {ex.Message}");
            }
        }

        /// <summary>
        /// Head 7: Execute item buy: 0=none, 1-4=faire (기사회생) options, 5-15=specific items
        /// </summary>
        private static void ExecuteItemBuy(int idx, JassUnit u, int itemBuy)
        {
            if (itemBuy == 0) return;

            try
            {
                JassPlayer player = Natives.Player(idx);

                if (itemBuy >= 1 && itemBuy <= 6)
                {
                    // Faire (기사회생) -- issue ability order on faire unit
                    int fairesEarned = _advCount[idx] / 7;
                    if (fairesEarned <= _faireUsed[idx])
                    {
                        Log($"[RLComm] Faire: p{idx} not available (adv={_advCount[idx]}, used={_faireUsed[idx]})");
                        return;
                    }

                    // Map idx to faire ability order strings
                    // 1=A04B(골드토큰,flare), 2=A04F(+3레벨,heal), 3=A0DD(+12스탯,sanctuary),
                    // 4=A04C(항마력,frenzy), 5=A04D(아이템+3000g,frostnova), 6=A00U(무적결계,holybolt)
                    string[] faireOrders = { "", "flare", "heal", "sanctuary", "frenzy", "frostnova", "holybolt" };
                    string orderStr = faireOrders[itemBuy];

                    // Find faire unit (h02Q or h02R) owned by this player
                    JassUnit faireUnit = default(JassUnit);
                    bool found = false;
                    try
                    {
                        JassPlayer jp = Natives.Player(idx);
                        JassGroup g = Natives.CreateGroup();
                        Natives.GroupEnumUnitsOfPlayer(g, jp, default(JassBooleanExpression));
                        JassUnit gu;
                        while ((gu = Natives.FirstOfGroup(g)).Handle != IntPtr.Zero)
                        {
                            int utid = (int)Natives.GetUnitTypeId(gu);
                            if (utid == FourCC("h02Q") || utid == FourCC("h02R"))
                            {
                                faireUnit = gu;
                                found = true;
                                break;
                            }
                            Natives.GroupRemoveUnit(g, gu);
                        }
                        Natives.DestroyGroup(g);
                    }
                    catch (Exception ex)
                    {
                        Log($"[RLComm] Faire unit search error p{idx}: {ex.Message}");
                    }

                    if (!found)
                    {
                        Log($"[RLComm] Faire: p{idx} no faire unit found (adv={_advCount[idx]})");
                        return;
                    }

                    // Issue the ability order on the faire unit
                    bool ok = false;
                    try
                    {
                        ok = Natives.IssueImmediateOrder(faireUnit, orderStr);
                    }
                    catch { }

                    if (ok)
                    {
                        _faireUsed[idx]++;
                        Log($"[RLComm] Faire: p{idx} used option {itemBuy} ({orderStr}) adv={_advCount[idx]} used={_faireUsed[idx]}");
                    }
                    else
                    {
                        Log($"[RLComm] Faire: p{idx} order failed ({orderStr})");
                    }
                    return;
                }
                else if (itemBuy >= 7 && itemBuy <= 17 && itemBuy < _shopItems.Length)
                {
                    ShopItem si = _shopItems[itemBuy];
                    int gold = GetPlayerGold(idx);
                    if (gold < si.cost)
                    {
                        Log($"[RLComm] ItemBuy: p{idx} not enough gold ({gold} < {si.cost})");
                        return;
                    }

                    // Check empty slot
                    if (!HasEmptyItemSlot(u))
                    {
                        Log($"[RLComm] ItemBuy: p{idx} no empty slot");
                        return;
                    }

                    // Deduct gold
                    SetPlayerGold(idx, gold - si.cost);

                    // Create item and give to hero
                    float ux = Natives.GetUnitX(u);
                    float uy = Natives.GetUnitY(u);
                    JassItem newItem = Natives.CreateItem((JassObjectId)FourCC(si.typeId), ux, uy);
                    if (newItem.Handle != IntPtr.Zero)
                    {
                        Natives.UnitAddItem(u, newItem);
                        Log($"[RLComm] ItemBuy: p{idx} bought {si.typeId} for {si.cost}g");
                    }
                }
            }
            catch (Exception ex)
            {
                Log($"[RLComm] ExecuteItemBuy error idx={idx}: {ex.Message}");
            }
        }

        /// <summary>
        /// Head 8: Execute item use: 0=none, 1-6=inventory slot
        /// </summary>
        private static void ExecuteItemUse(int idx, JassUnit u, int itemUse,
            int unitTarget, float pointX, float pointY, float cx, float cy)
        {
            if (itemUse == 0) return;

            try
            {
                int slot = itemUse - 1; // convert to 0-based
                if (slot < 0 || slot >= 6) return;

                JassItem itm = Natives.UnitItemInSlot(u, slot);
                if (itm.Handle == IntPtr.Zero) return;

                int itmType = (int)Natives.GetItemTypeId(itm);
                if (itmType == 0) return;

                // Try point use if point mode selected
                if (unitTarget == 13)
                {
                    float tx, ty;
                    DecodePolarPoint(pointX, pointY, 600f, cx, cy, out tx, out ty);
                    Natives.UnitUseItemPoint(u, itm, tx, ty);
                }
                else if (unitTarget >= 0 && unitTarget <= 12)
                {
                    int myTeamBase = idx < 6 ? 0 : 6;
                    int enemyTeamBase = idx < 6 ? 6 : 0;
                    JassUnit target = ResolveTargetUnit(idx, unitTarget, myTeamBase, enemyTeamBase);
                    if (target.Handle != IntPtr.Zero)
                    {
                        Natives.UnitUseItemTarget(u, itm, target);
                    }
                    else
                    {
                        Natives.UnitUseItem(u, itm);
                    }
                }
                else
                {
                    // Default: immediate use
                    Natives.UnitUseItem(u, itm);
                }

                Log($"[RLComm] ItemUse: p{idx} slot={slot} type={TypeIdToString(itmType)}");
            }
            catch (Exception ex)
            {
                Log($"[RLComm] ExecuteItemUse error idx={idx}: {ex.Message}");
            }
        }

        /// <summary>
        /// Head 9: Execute seal use: 0=none, 1=first_activate, 2=cd_reset, 3=hp_recover, 4=mp_recover, 5=revive, 6=teleport
        /// </summary>
        // Order strings for command seal abilities on the grail unit.
        // These trigger JASS s__CommandSeal_onCast which handles charges, cooldowns,
        // stat point costs, and the actual effects (HP/MP recover, revive, teleport, etc.)
        private static readonly string[] SEAL_ORDER_STRINGS = {
            "",                  // 0: unused
            "carrionscarabsoff", // 1: First Activate  (A094, ANcl)
            "blackarrowoff",     // 2: CD Reset        (A043, AAns)
            "coldarrows",        // 3: HP Recover      (A044, AAns)
            "doom",              // 4: MP Recover      (A05Q, ANcl)
            "curse",             // 5: Revive          (A0DU, ANcl)
            "curseoff",          // 6: Teleport        (A0DY, ANcl, point target)
        };

        private static void ExecuteSealUse(int idx, JassUnit u, int sealUse, float pointX, float pointY)
        {
            if (sealUse == 0) return;
            if (sealUse < 1 || sealUse > 6) return;

            try
            {
                // Validate: grail unit must exist
                if (!_grailRegistered[idx])
                {
                    Log($"[RLComm] SealUse: p{idx} no grail registered");
                    return;
                }

                // Validate: check seal charges (prevent wasting actions)
                var sealData = GetSealItem(idx);
                int sealCharges = sealData.charges;

                bool isFirstActive = JassStateCache.firstActive[idx] != 0;
                int cost = isFirstActive ? _sealCostsFirstActive[sealUse] : _sealCosts[sealUse];

                if (sealCharges < cost)
                {
                    Log($"[RLComm] SealUse: p{idx} not enough charges ({sealCharges} < {cost})");
                    return;
                }

                // Validate: check cooldown (skip for first activate)
                if (JassStateCache.sealCd[idx] > 0 && sealUse != 1)
                {
                    Log($"[RLComm] SealUse: p{idx} on cooldown ({JassStateCache.sealCd[idx]})");
                    return;
                }

                // Issue order on grail unit — triggers JASS s__CommandSeal_onCast
                // which handles charge deduction, stat point cost, cooldowns, and effects.
                JassUnit grail = _grailUnits[idx];
                string orderStr = SEAL_ORDER_STRINGS[sealUse];
                bool ok;

                if (sealUse == 6) // Teleport — point target (A0DY, Ncl2=2)
                {
                    float tx = MAP_MIN_X + (pointX + 1f) * 0.5f * (MAP_MAX_X - MAP_MIN_X);
                    float ty = MAP_MIN_Y + (pointY + 1f) * 0.5f * (MAP_MAX_Y - MAP_MIN_Y);
                    tx = Clampf(tx, MAP_MIN_X + 64, MAP_MAX_X - 64);
                    ty = Clampf(ty, MAP_MIN_Y + 64, MAP_MAX_Y - 64);
                    ok = Natives.IssuePointOrder(grail, orderStr, tx, ty);
                    Log($"[RLComm] SealUse: p{idx} teleport order={orderStr} to ({tx:F0},{ty:F0}) ok={ok}");
                }
                else // All other seals — immediate order
                {
                    ok = Natives.IssueImmediateOrder(grail, orderStr);
                    Log($"[RLComm] SealUse: p{idx} type={sealUse} order={orderStr} ok={ok}");
                }

                // After JASS processes CD reset, also clear our local cooldown tracker
                if (ok && sealUse == 2)
                {
                    _cdTracker.ResetAll(idx);
                }
            }
            catch (Exception ex)
            {
                Log($"[RLComm] ExecuteSealUse error idx={idx}: {ex.Message}");
            }
        }

        /// <summary>
        /// Head 10: Execute faire (gold) send: 0=none, 1-5=ally (relative index within team)
        /// Transfers FAIRE_TRANSFER_AMOUNT gold to the specified teammate.
        /// </summary>
        private static void ExecuteFaireSend(int idx, int faireSend)
        {
            if (faireSend == 0) return;
            if (faireSend < 1 || faireSend > 5) return;

            try
            {
                int senderGold = GetPlayerGold(idx);
                if (senderGold < FAIRE_TRANSFER_AMOUNT)
                {
                    Log($"[RLComm] FaireSend: p{idx} not enough gold ({senderGold} < {FAIRE_TRANSFER_AMOUNT})");
                    return;
                }

                int allyPid = ResolveAllyIndex(idx, faireSend);
                if (allyPid < 0 || !heroRegistered[allyPid])
                {
                    Log($"[RLComm] FaireSend: p{idx} invalid ally index {faireSend}");
                    return;
                }

                if (!IsUnitAlive(heroes[allyPid]))
                {
                    Log($"[RLComm] FaireSend: p{idx} ally p{allyPid} is dead");
                    return;
                }

                int receiverGold = GetPlayerGold(allyPid);
                int cappedAmount = Math.Min(FAIRE_TRANSFER_AMOUNT, FAIRE_CAP - receiverGold);
                if (cappedAmount <= 0)
                {
                    Log($"[RLComm] FaireSend: p{idx} receiver p{allyPid} at faire cap");
                    return;
                }

                SetPlayerGold(idx, senderGold - FAIRE_TRANSFER_AMOUNT);
                SetPlayerGold(allyPid, receiverGold + cappedAmount);

                Log($"[RLComm] FaireSend: p{idx} -> p{allyPid} {cappedAmount}g");
            }
            catch (Exception ex)
            {
                Log($"[RLComm] ExecuteFaireSend error idx={idx}: {ex.Message}");
            }
        }

        /// <summary>
        /// Head 11: Execute faire request: 0=none, 1-5=ally index
        /// Registers a pending request for the target ally.
        /// </summary>
        private static void ExecuteFaireRequest(int idx, int faireRequest)
        {
            if (faireRequest == 0) return;
            if (faireRequest < 1 || faireRequest > 5) return;

            try
            {
                int allyPid = ResolveAllyIndex(idx, faireRequest);
                if (allyPid < 0 || !heroRegistered[allyPid]) return;

                // Register request: target ally receives a request from idx
                _faireRequests[allyPid] = new FaireRequest { requester = idx, amount = FAIRE_TRANSFER_AMOUNT };
                Log($"[RLComm] FaireRequest: p{idx} requests {FAIRE_TRANSFER_AMOUNT}g from p{allyPid}");
            }
            catch (Exception ex)
            {
                Log($"[RLComm] ExecuteFaireRequest error idx={idx}: {ex.Message}");
            }
        }

        /// <summary>
        /// Head 12: Execute faire respond: 0=none, 1=accept, 2=deny
        /// </summary>
        private static void ExecuteFaireRespond(int idx, int faireRespond)
        {
            if (faireRespond == 0) return;

            try
            {
                if (!_faireRequests.ContainsKey(idx))
                {
                    Log($"[RLComm] FaireRespond: p{idx} no pending request");
                    return;
                }

                var req = _faireRequests[idx];
                _faireRequests.Remove(idx);

                if (faireRespond == 1) // accept
                {
                    int myGold = GetPlayerGold(idx);
                    if (myGold < req.amount)
                    {
                        Log($"[RLComm] FaireRespond: p{idx} accept but not enough gold ({myGold} < {req.amount})");
                        return;
                    }

                    int requesterGold = GetPlayerGold(req.requester);
                    int cappedAmount = Math.Min(req.amount, FAIRE_CAP - requesterGold);
                    if (cappedAmount <= 0) return;

                    SetPlayerGold(idx, myGold - req.amount);
                    SetPlayerGold(req.requester, requesterGold + cappedAmount);

                    Log($"[RLComm] FaireRespond: p{idx} accepted -> p{req.requester} {cappedAmount}g");
                }
                else // deny
                {
                    Log($"[RLComm] FaireRespond: p{idx} denied request from p{req.requester}");
                }
            }
            catch (Exception ex)
            {
                Log($"[RLComm] ExecuteFaireRespond error idx={idx}: {ex.Message}");
            }
        }

        // ============================================================
        // Episode Done Check
        // ============================================================

        /// <summary>
        /// Check if the episode is done.
        /// - episodeState != 0 (team wipe / score reached) -> true
        /// - Score-based termination
        /// - tickCount > 18000 (30 min @ 10/s) -> timeout, return true
        /// </summary>
        private static bool IsEpisodeDone()
        {
            if (episodeState != 0) return true;

            // Score-based termination
            if (JassStateCache.teamScore[0] >= TARGET_SCORE)
            {
                episodeState = 4; // score reached
                return true;
            }
            if (JassStateCache.teamScore[1] >= TARGET_SCORE)
            {
                episodeState = 4;
                return true;
            }

            // No tick-based timeout — let episodes run until natural end
            // (team wipe or score reached)

            return false;
        }

        /// <summary>
        /// Build the DONE message JSON with extended info.
        /// </summary>
        private static string BuildDoneJson()
        {
            float gameTime = tickCount * TICK_INTERVAL;

            // Determine winner
            string winner = "draw";
            string reason = "unknown";
            if (episodeState == 1) { winner = "team1"; reason = "team_wipe"; } // team0 wiped = team1 wins
            else if (episodeState == 2) { winner = "team0"; reason = "team_wipe"; } // team1 wiped = team0 wins
            else if (episodeState == 3)
            {
                reason = "timeout";
                if (JassStateCache.teamScore[0] > JassStateCache.teamScore[1]) winner = "team0";
                else if (JassStateCache.teamScore[1] > JassStateCache.teamScore[0]) winner = "team1";
                else winner = "draw";
            }
            else if (episodeState == 4)
            {
                reason = "score";
                winner = JassStateCache.teamScore[0] >= TARGET_SCORE ? "team0" : "team1";
            }

            var doneObj = new JObject
            {
                ["tick"] = tickCount,
                ["winner"] = winner,
                ["score_ally"] = JassStateCache.teamScore[0],
                ["score_enemy"] = JassStateCache.teamScore[1],
                ["game_time"] = (float)Math.Round(gameTime, 1),
                ["reason"] = reason,
                ["episode_state"] = episodeState
            };

            return doneObj.ToString(Formatting.None);
        }

        // ============================================================
        // OnTick -- Non-blocking UDP send STATE / recv ACTION
        // ============================================================

        private static void OnTick()
        {
            if (!heroesReady)
            {
                if (tickCount == 0)
                    Log($"[RLComm] OnTick but heroes not ready ({heroCount}/{MAX_PLAYERS})");
                return;
            }

            tickCount++;


            // Wall-clock tick rate using direct kernel32 (always real time)
            {
                uint wallNow = RealGetTickCount();
                if (!_firstTickWallSet)
                {
                    _firstTickWallMs = wallNow;
                    _lastTickLogWallMs = wallNow;
                    _firstTickWallSet = true;
                }
                if (tickCount % 50 == 0)
                {
                    uint last50ms = wallNow - _lastTickLogWallMs;
                    uint totalMs = wallNow - _firstTickWallMs;
                    double ticksPerSec = last50ms > 0 ? 50000.0 / last50ms : 0;
                    Log($"[RLComm] TickRate: tick={tickCount} last50={last50ms}ms total={totalMs}ms rate={ticksPerSec:F1}/s");
                    _lastTickLogWallMs = wallNow;
                }
                // Force GC every 500 ticks to prevent OOM under Wine
                if (tickCount % 500 == 0)
                {
                    GC.Collect(0, GCCollectionMode.Optimized);
                }
            }

            // Re-apply speed patch every 100 ticks (in case game resets interval)
            if (tickCount % 100 == 0)
                ReapplySpeedPatch();

            if (tickCount == 1)
                Log("[RLComm] First tick! Starting UDP state transmission");

            // Check episode done
            if (IsEpisodeDone())
            {
                try
                {
                    byte[] donePkt = BuildDoneBinary();
                    // Send DONE multiple times to ensure delivery (UDP is unreliable)
                    for (int i = 0; i < 3; i++)
                    {
                        udpSend.Send(donePkt, donePkt.Length, inferenceEndpoint);
                        System.Threading.Thread.Sleep(50);
                    }
                    Log($"[RLComm] Episode done sent via UDP (3x), tick={tickCount}");
                }
                catch (Exception ex)
                {
                    Log($"[RLComm] DONE send failed: {ex.Message}");
                }
                TriggerRestart();
                return;
            }

            try
            {
                // 1. Build and send STATE (fire-and-forget, non-blocking)
                byte[] statePkt = BuildStateBinary();
                udpSend.Send(statePkt, statePkt.Length, inferenceEndpoint);

                // 2. Non-blocking recv ACTION (drain all, use latest)
                byte[] latestAction = null;
                while (udpRecv.Available > 0)
                {
                    IPEndPoint remoteEP = null;
                    latestAction = udpRecv.Receive(ref remoteEP);
                }

                // 3. Apply new ACTION if received (1 time only, no caching)
                if (latestAction != null && latestAction.Length >= 8)
                    ProcessActionBinary(latestAction);

                // Clear alarm states
                for (int i = 0; i < MAX_PLAYERS; i++)
                    _alarmState[i] = false;
            }
            catch (Exception ex)
            {
                Log($"[RLComm] OnTick UDP error: {ex.Message}");
            }
        }

        // ============================================================
        // Restart -- terminate war3.exe so Docker entrypoint restarts it
        // ============================================================

        private static void TriggerRestart()
        {
            Log("[RLComm] TriggerRestart: writing sentinel + ExitProcess");

            // Write sentinel file for entrypoint.sh to detect and kill wineserver
            // Wine maps Z:\ to /, so Z:\tmp\rl_episode_done = /tmp/rl_episode_done
            try { System.IO.File.WriteAllText(@"Z:\tmp\rl_episode_done", "done"); }
            catch (Exception ex) { Log($"[RLComm] Sentinel write failed: {ex.Message}"); }

            // ExitProcess may not work under Wine's injected CLR, but try anyway
            ExitProcess(0);
        }

        // (ServerLoop removed -- replaced by UDP protocol)

        // ============================================================
        // Plugin Interface
        // ============================================================

        // Direct file logging (avoids Trace listener issues)
        private static string _logFilePath;
        private static string _debugLogPath;
        private static void Log(string msg)
        {
            string line = $"[{DateTime.Now:HH:mm:ss.fff}] {msg}\r\n";
            System.Diagnostics.Trace.WriteLine(msg);
            if (_debugLogPath != null)
                try { File.AppendAllText(_debugLogPath, line); } catch { }
        }

        public void Initialize()
        {
            // Setup direct file logging — use assembly dir (works in Docker + Windows)
            try
            {
                string logDir = Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location) ?? ".";
                _logFilePath = Path.Combine(logDir, "rlcomm_init.log");
                _debugLogPath = Path.Combine(logDir, "rlcomm_debug.log");
                File.WriteAllText(_logFilePath, $"[RLComm] Init started at {DateTime.Now}, asm={System.Reflection.Assembly.GetExecutingAssembly().Location}\n");
            }
            catch (Exception ex)
            {
                // Fallback: try War3Client root
                try
                {
                    string fallbackDir = AppDomain.CurrentDomain.BaseDirectory;
                    _logFilePath = Path.Combine(fallbackDir, "rlcomm_init.log");
                    _debugLogPath = Path.Combine(fallbackDir, "rlcomm_debug.log");
                    File.WriteAllText(_logFilePath, $"[RLComm] Init started (fallback) at {DateTime.Now}, err={ex.Message}\n");
                }
                catch { }
            }

            // No TextWriterTraceListener - it locks the file and blocks File.AppendAllText
            Trace.AutoFlush = true;

            Natives.Add(new PreloaderDelegate(PreloaderOverride), "Preloader", "(S)V");
            Natives.Add(new RLRandDelegate(RLRand), "RLRand", "(II)I");
            Natives.Add(new GetRandomIntDelegate(GetRandomIntOverride), "GetRandomInt", "(II)I");
            Natives.Add(new GetRandomRealDelegate(GetRandomRealOverride), "GetRandomReal", "(RR)R");
            Natives.Add(new RLSetPDATDelegate(RLSetPDAT), "RLSetPDAT", "(IIIIIIIIIIIIII)V");
            Natives.Add(new RLSetScoreDelegate(RLSetScore), "RLSetScore", "(II)V");
            Natives.Add(new RLTickDelegate(RLTickNative), "RLTick", "(I)V");
            Natives.Add(new RLKillDelegate(RLKillNative), "RLKill", "(II)V");
            Natives.Add(new RLCreepDelegate(RLCreepNative), "RLCreep", "(I)V");
            Natives.Add(new RLLvUpDelegate(RLLvUpNative), "RLLvUp", "(II)V");
            Natives.Add(new RLPortalDelegate(RLPortalNative), "RLPortal", "(I)V");
            Natives.Add(new RLAlarmDelegate(RLAlarmNative), "RLAlarm", "(I)V");
            Natives.Add(new RLDoneDelegate(RLDoneNative), "RLDone", "(I)V");
            Log($"[RLComm] All natives registered (RNG seed={Environment.TickCount})");
        }

        public void OnGameLoad()
        {
            Log("[RLComm] >>> OnGameLoad() called <<<");
            running = true;
            InitHeroDataTable();

            // Reset all state for new game
            tickCount = 0;
            episodeState = 0;
            heroCount = 0;
            heroesReady = false;
            _prevPosValid = false;
            _pathabilityComputed = false;
            _pathabilityGrid = null;
            _firstTickWallSet = false;
            _fogInitialized = false;
            _fogInitFailed = false;
            _fogBitmapPtr = IntPtr.Zero;
            _eventQueue.Clear();
            _cdTracker.Clear();
            _faireRequests.Clear();
            JassStateCache.Reset();
            for (int i = 0; i < MAX_PLAYERS; i++)
            {
                heroRegistered[i] = false;
                heroHandleIds[i] = 0;
                _grailRegistered[i] = false;
                _alarmState[i] = false;
                _advCount[i] = 0;
                _faireUsed[i] = 0;
                _prevX[i] = 0f;
                _prevY[i] = 0f;
                for (int s = 0; s < 5; s++) JassStateCache.attrAcquired[i, s] = 0;
            }

            // ---- UDP setup ----
            try
            {
                string host = Environment.GetEnvironmentVariable("INFERENCE_HOST") ?? "127.0.0.1";
                int udpPort = int.TryParse(Environment.GetEnvironmentVariable("INFERENCE_PORT"), out int p) ? p : sendPort;
                int recvP = int.TryParse(Environment.GetEnvironmentVariable("RL_RECV_PORT"), out int rp) ? rp : recvPort;

                inferenceEndpoint = new IPEndPoint(IPAddress.Parse(host), udpPort);
                udpSend = new UdpClient();
                udpRecv = new UdpClient(recvP);
                udpRecv.Client.Blocking = false;

                Log($"[RLComm] UDP initialized: send={host}:{udpPort}, recv=0.0.0.0:{recvP}");
            }
            catch (Exception ex)
            {
                Log($"[RLComm] UDP init error: {ex.Message}");
            }

            // ---- Speed acceleration (NO timing API hooks!) ----
            string speedEnv = Environment.GetEnvironmentVariable("WC3_SPEED_MULTIPLIER");
            Log($"[RLComm] WC3_SPEED_MULTIPLIER env = '{speedEnv}'");
            if (!string.IsNullOrEmpty(speedEnv) && double.TryParse(speedEnv, out double mult) && mult > 1.0)
            {
                _speedMultiplier = mult;
                int hooked = 0;

                // 1. Hook SetTimer -- make WM_TIMER fire N times faster
                IntPtr user32 = Kernel32.GetModuleHandleA("user32.dll");
                if (user32 != IntPtr.Zero)
                {
                    IntPtr stAddr = Kernel32.GetProcAddress(user32, "SetTimer");
                    if (stAddr != IntPtr.Zero)
                    {
                        _origSetTimer = Memory.InstallHook<SetTimerDelegate>(
                            stAddr, new SetTimerDelegate(SetTimerHook), true, false);
                        hooked++;
                        Log($"[RLComm] Speed: SetTimer hooked at 0x{stAddr.ToInt64():X8}");
                    }
                }

                // 2. Patch .rdata speed table
                PatchSpeedTable();

                // 3. Set timer resolution to 1ms
                timeBeginPeriod(1);
                Log("[RLComm] Speed: timeBeginPeriod(1) set");

                Log($"[RLComm] Speed: {hooked} hook(s), multiplier={mult}x");
                Log("[RLComm] Speed: JASS must call RL_SCAN1 -> SetGameSpeed(Normal) -> RL_SCAN2 -> SetGameSpeed(Fastest) -> RL_SPEEDPATCH");
            }
            else
            {
                Log("[RLComm] Speed: WC3_SPEED_MULTIPLIER not set or <=1, normal speed");
            }

            // 4. Loading screen -- poll-based keybd_event (Docker/Wine only)
            string noDismiss = Environment.GetEnvironmentVariable("RL_NO_DISMISS");
            if (noDismiss == null || noDismiss != "1")
                DismissLoadingLoop();
            else
                Log("[RLComm] DismissLoading: DISABLED (RL_NO_DISMISS=1)");

            // Speed monitor thread
            if (_speedMultiplier > 1.0)
            {
                Thread monitorThread = new Thread(() =>
                {
                    Thread.Sleep(5000);
                    while (running)
                    {
                        Thread.Sleep(10000);
                        long c6 = Interlocked.Exchange(ref _cntSetTimer, 0);
                        long c7 = Interlocked.Exchange(ref _cntLoopTick, 0);
                        Log($"[RLComm] SpeedStats/10s: loopTick={c7} settimer={c6}");
                    }
                })
                { IsBackground = true, Name = "RLCommSpeedMon" };
                monitorThread.Start();
            }

            // Watchdog thread: if OnTick stops being called for 30s (game ended/stuck),
            // create sentinel file to trigger episode restart.
            // This handles the case where WC3 reaches victory screen and destroys timers.
            {
                Thread watchdogThread = new Thread(() =>
                {
                    Thread.Sleep(60000); // Wait 60s for game to start properly
                    int lastTick = tickCount;
                    int staleCount = 0;
                    while (running)
                    {
                        Thread.Sleep(5000); // Check every 5 seconds
                        int curTick = tickCount;
                        if (curTick > 0 && curTick == lastTick)
                        {
                            staleCount++;
                            if (staleCount >= 6) // 6 * 5s = 30 seconds stale
                            {
                                Log($"[RLComm] Watchdog: OnTick stale for 30s (tick={curTick}), forcing restart");
                                TriggerRestart();
                                break;
                            }
                        }
                        else
                        {
                            staleCount = 0;
                        }
                        lastTick = curTick;
                    }
                })
                { IsBackground = true, Name = "RLCommWatchdog" };
                watchdogThread.Start();
                Log("[RLComm] Watchdog thread started (30s stale threshold)");
            }
        }

        public void OnMapLoad()
        {
            _mapLoaded = true;
            Log("[RLComm] Map loaded");
        }

        private static volatile bool _loadingDismissRunning;
        private static volatile bool _mapLoaded;

        /// <summary>
        /// Poll-based loading screen dismissal.
        /// Sends VK_SPACE via keybd_event every 500ms until heroesReady or 60s timeout.
        /// </summary>
        private static void DismissLoadingLoop()
        {
            _loadingDismissRunning = true;
            Thread dismissThread = new Thread(() =>
            {
                try
                {
                    // Wait for WC3 window to exist
                    IntPtr hwnd = IntPtr.Zero;
                    for (int r = 0; r < 30 && hwnd == IntPtr.Zero; r++)
                    {
                        Thread.Sleep(500);
                        hwnd = FindWindowA("Warcraft III", null);
                    }
                    if (hwnd == IntPtr.Zero)
                    {
                        Log("[RLComm] DismissLoading: window not found after 15s, giving up");
                        return;
                    }

                    Log($"[RLComm] DismissLoading: hwnd=0x{hwnd.ToInt64():X}, polling start...");

                    // Wait 10s for JASS main() to finish, then use keybd_event
                    Log("[RLComm] DismissLoading: waiting 10s for JASS main()...");
                    Thread.Sleep(10000);

                    Log("[RLComm] DismissLoading: starting keybd_event loop...");
                    for (int i = 0; i < 120; i++) // 120 x 500ms = 60s max
                    {
                        if (heroesReady)
                        {
                            Log($"[RLComm] DismissLoading: heroes ready after {(i * 500) + 10000}ms, stopping");
                            break;
                        }

                        bool fgResult = SetForegroundWindow(hwnd);
                        if (i % 10 == 0)
                            Log($"[RLComm] DismissLoading: attempt {i}, SetForeground={fgResult}, sending VK_SPACE");

                        keybd_event(0x20, 0, 0, UIntPtr.Zero); // VK_SPACE down
                        Thread.Sleep(30);
                        keybd_event(0x20, 0, 2, UIntPtr.Zero); // VK_SPACE up (KEYEVENTF_KEYUP=2)
                        Thread.Sleep(470);
                    }

                    if (!heroesReady)
                        Log("[RLComm] DismissLoading: timeout (60s), game may not have started");
                }
                catch (Exception ex)
                {
                    Log($"[RLComm] DismissLoading error: {ex.Message}");
                }
                finally
                {
                    _loadingDismissRunning = false;
                }
            })
            { IsBackground = true, Name = "RLCommDismissLoading" };
            dismissThread.Start();
        }

        public void OnMapEnd()
        {
            Log("[RLComm] Map ended, resetting state");
            heroesReady = false;
            heroCount = 0;
            _prevPosValid = false;
            _pathabilityComputed = false;
            _eventQueue.Clear();
            _cdTracker.Clear();
            _faireRequests.Clear();
            for (int i = 0; i < MAX_PLAYERS; i++)
            {
                heroRegistered[i] = false;
                heroHandleIds[i] = 0;
                _grailRegistered[i] = false;
                _alarmState[i] = false;
                for (int s = 0; s < 5; s++) JassStateCache.attrAcquired[i, s] = 0;
            }
        }

        public void OnProgramExit()
        {
            Log("[RLComm] Program exit, closing UDP");
            running = false;
            try { udpSend?.Close(); } catch { }
            try { udpRecv?.Close(); } catch { }
        }
    }
}
