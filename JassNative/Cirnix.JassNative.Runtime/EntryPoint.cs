using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Threading;
using System.Windows;

using Cirnix.JassNative.Runtime.Utilities;
using Cirnix.JassNative.Runtime.Windows;

using EasyHook;

namespace Cirnix.JassNative.Runtime
{
    public unsafe class EntryPoint : IEntryPoint
    {
        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern IntPtr OpenEventA(uint dwDesiredAccess, bool bInheritHandle, string lpName);

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool SetEvent(IntPtr hEvent);

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool CloseHandle(IntPtr hObject);

        private const uint EVENT_MODIFY_STATE = 0x0002;

        [UnmanagedFunctionPointer(CallingConvention.ThisCall)]
        private delegate IntPtr Unknown__SetStateDelegate(IntPtr @this, bool endMap, bool endEngine);

        private Unknown__SetStateDelegate Unknown__SetState;

        private Kernel32.LoadLibraryAPrototype LoadLibraryA;

        // Flat layout: all DLLs in hackPath, no subfolders

        public bool IsInMap { get; private set; }

        public EntryPoint(RemoteHooking.IContext hookingContext, bool isDebugging, string hackPath, string installPath)
        {
            try
            {
            }
            catch (Exception exception)
            {
                MessageBox.Show(
                    "Fatal exception!" + Environment.NewLine +
                    exception + Environment.NewLine +
                    "Aborting execution!",
                    GetType() + ".ctor(...)", MessageBoxButton.OK, MessageBoxImage.Error);
                Process.GetCurrentProcess().Kill();
            }
        }

        public void Run(RemoteHooking.IContext hookingContext, bool isDebugging, string hackPath, string installPath)
        {
            try
            {
                PluginSystem.HackPath = hackPath;
                PluginSystem.IsDebugging = isDebugging;
                if (isDebugging)
                {
                    DebuggerApplication.Start(hackPath);
                    while (!DebuggerApplication.IsReady)
                        Thread.Sleep(1); // Sleep(0) is a nono.
                }
                // Everything traced will be written to "debug.log".
                string logPath = Path.Combine(hackPath, "logs\\runtime");
                if (!Directory.Exists(logPath))
                    Directory.CreateDirectory(logPath);
                Trace.Listeners.Add(new TextWriterTraceListener(Path.Combine(logPath, $"{DateTime.Now:yyyy-MM-dd HH.mm.ss}.log")));
                Trace.IndentSize = 2;

                // We autoflush our trace, so we get everything immediately. This
                // makes tracing a bit more expensive, but means we still get a log
                // even if there's a fatal crash.
                Trace.AutoFlush = true;

                Trace.WriteLine("-------------------");
                Trace.WriteLine(DateTime.Now);
                Trace.WriteLine("-------------------");

                AppDomain.CurrentDomain.AssemblyResolve += (object sender, ResolveEventArgs args) =>
                {
                    var path = string.Empty;
                    // extract the file name
                    var file = string.Empty;
                    if (args.Name.IndexOf(',') >= 0)
                        file = args.Name.Substring(0, args.Name.IndexOf(',')) + ".dll";
                    else if (args.Name.IndexOf(".dll") >= 0)
                        file = Path.GetFileName(args.Name);
                    else
                        return null;

                    // Flat layout: search hackPath only
                    path = Directory.GetFiles(hackPath, file, SearchOption.TopDirectoryOnly).FirstOrDefault();
                    if (!string.IsNullOrEmpty(path))
                        try
                        {
                            return Assembly.LoadFrom(path);
                        }
                        catch
                        {
                            return Assembly.Load(File.ReadAllBytes(path));
                        }

                    return null;
                };

                AppDomain.CurrentDomain.ReflectionOnlyAssemblyResolve += (object sender, ResolveEventArgs args) =>
                {
                    var file = string.Empty;
                    if (args.Name.IndexOf(',') >= 0)
                        try { return Assembly.ReflectionOnlyLoad(args.Name); }
                        catch { file = args.Name.Substring(0, args.Name.IndexOf(',')) + ".dll"; }
                    else if (args.Name.IndexOf(".dll") >= 0)
                        file = Path.GetFileName(args.Name);
                    else
                        return null;

                    // Flat layout: search hackPath only
                    var path = Directory.GetFiles(hackPath, file, SearchOption.TopDirectoryOnly).FirstOrDefault();
                    if (!string.IsNullOrEmpty(path))
                        try { return Assembly.ReflectionOnlyLoadFrom(path); }
                        catch { return Assembly.ReflectionOnlyLoad(File.ReadAllBytes(path)); }
                    return null;
                };

                var sw = new Stopwatch();

                Trace.WriteLine("Preparing folders . . . ");
                Trace.Indent();
                sw.Restart();
                sw.Stop();
                Trace.WriteLine($"Install Path: {installPath}");
                Trace.WriteLine($"Hack Path:    {hackPath}");
                Trace.WriteLine($"Done! ({sw.Elapsed.TotalMilliseconds:0.00} ms)");
                Trace.Unindent();

                Trace.WriteLine($"Loading plugins from '{hackPath}' . . .");
                Trace.Indent();
                sw.Restart();
                PluginSystem.InitSystem();
                PluginSystem.LoadPlugins(hackPath);
                sw.Stop();
                Trace.WriteLine($"Done! ({sw.Elapsed.TotalMilliseconds:0.00} ms)");
                Trace.Unindent();

                bool isProxyMode = Environment.GetEnvironmentVariable("JNLOADER_PROXY_MODE") == "1";

                if (isProxyMode)
                {
                    // Proxy mode (Wine/Docker): avoid EasyHook LocalHook on kernel32 functions.
                    // Wine's kernel32.dll doesn't tolerate inline code patching.
                    // Instead, poll for game.dll with GetModuleHandle.
                    Trace.WriteLine("Proxy mode: polling for game.dll (no LocalHook)...");

                    new Thread(() =>
                    {
                        try
                        {
                            IntPtr gameDll = IntPtr.Zero;
                            while (gameDll == IntPtr.Zero)
                            {
                                gameDll = Kernel32.GetModuleHandleA("game.dll");
                                if (gameDll == IntPtr.Zero)
                                    Thread.Sleep(200);
                            }

                            Trace.WriteLine($"game.dll detected at 0x{gameDll.ToInt32():X8}");
                            PluginSystem.OnGameLoad();

                            // Hook Unknown__SetState via LocalHook on game.dll (NOT kernel32)
                            try
                            {
                                Unknown__SetState = Memory.InstallHook(
                                    gameDll + Addresses.Unknown__SetStateOffset,
                                    new Unknown__SetStateDelegate(Unknown__SetStateHook), true, false);
                                Trace.WriteLine("Unknown__SetState hook installed.");
                            }
                            catch (Exception hookEx)
                            {
                                Trace.WriteLine("WARNING: Unknown__SetState hook failed: " + hookEx.Message);
                                Trace.WriteLine("Map lifecycle events will not be detected.");
                            }

                            // Signal proxy DLL that all hooks are installed.
                            // ijlInit() is waiting on this event before returning to game.dll.
                            try
                            {
                                IntPtr hEvent = OpenEventA(EVENT_MODIFY_STATE, false, "JNProxyHooksReady");
                                if (hEvent != IntPtr.Zero)
                                {
                                    SetEvent(hEvent);
                                    CloseHandle(hEvent);
                                    Trace.WriteLine("[Proxy] Signaled JNProxyHooksReady event");
                                }
                            }
                            catch { }
                        }
                        catch (Exception ex)
                        {
                            Trace.WriteLine("game.dll polling error: " + ex);
                        }
                    }) { IsBackground = true }.Start();

                    Trace.WriteLine("Sleep Proceed! (proxy mode)");
                    Thread.Sleep(Timeout.Infinite);
                }
                else
                {
                    // Default mode: use EasyHook LocalHook on LoadLibraryA
                    LoadLibraryA = Memory.InstallHook(LocalHook.GetProcAddress("kernel32.dll", "LoadLibraryA"), new Kernel32.LoadLibraryAPrototype(LoadLibraryAHook), false, true);

                    // Everyone has had their chance to inject stuff,
                    // time to wake up the process.
                    try
                    {
                        RemoteHooking.WakeUpProcess();
                        Trace.WriteLine("WakeUpProcess Proceed!");
                    }
                    catch (Exception)
                    {
                        Trace.WriteLine("WakeUpProcess skipped.");
                    }
                    // Let the thread stay alive, so all hooks stay alive as well.
                    Trace.WriteLine("Sleep Proceed!");
                    Thread.Sleep(Timeout.Infinite);
                }
            }
            catch (Exception exception)
            {
                MessageBox.Show(
                    "Fatal exception!" + Environment.NewLine +
                    exception + Environment.NewLine +
                    "Aborting execution!",
                    GetType() + ".Run(...)", MessageBoxButton.OK, MessageBoxImage.Error);
                Process.GetCurrentProcess().Kill();
            }
        }

        private IntPtr LoadLibraryAHook(string fileName)
        {
            IntPtr module = LoadLibraryA(fileName);

            switch (fileName.ToLower())
            {
                case "game.dll":
                    PluginSystem.OnGameLoad();

                    // Prepare the Unknown__SetState hook.
                    Unknown__SetState = Memory.InstallHook(module + Addresses.Unknown__SetStateOffset, new Unknown__SetStateDelegate(Unknown__SetStateHook), true, false);

                    break;
            }

            return module;
        }

        private IntPtr Unknown__SetStateHook(IntPtr @this, bool endMap, bool endEngine)
        {
            try
            {
                if (endMap || IsInMap)
                {
                    IsInMap = false;
                    PluginSystem.OnMapEnd();
                    if (endEngine)
                        PluginSystem.OnProgramExit();
                }
                else
                {
                    IsInMap = true;
                    PluginSystem.OnMapLoad();
                }
            }
            catch (Exception e)
            {
                Trace.WriteLine("Unhandled Exception in " + typeof(EntryPoint).Name + ".Unknown__SetStateHook!");
                Trace.WriteLine(e.ToString());
            }

            return Unknown__SetState(@this, endMap, endEngine);
        }
    }
}
