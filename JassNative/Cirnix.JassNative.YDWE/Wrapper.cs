using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;

using Cirnix.JassNative.Runtime;
using Cirnix.JassNative.Runtime.Windows;

namespace Cirnix.JassNative.YDWE
{
    internal static class Wrapper
    {
        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        private delegate void InitializePrototype();
        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        private delegate bool SetWindowPrototype(IntPtr hWnd);

        private static Dictionary<string, IntPtr> plugins = new Dictionary<string, IntPtr>();

        private static IntPtr LoadLibraryA(string path)
        {
            IntPtr BaseAddr = Kernel32.LoadLibrary(path);
            if (BaseAddr == IntPtr.Zero)
                Trace.WriteLine($"{path} Load Failed!");
            else
                Trace.WriteLine($"0x{BaseAddr.ToString("X8")} : {path} Loaded!");
            return BaseAddr;
        }

        private static readonly string[] binDlls = { "luacore.dll", "ydbase.dll", "SlkLib.dll" };
        private static readonly string[] pluginDlls = { "yd_jass_api.dll", "yd_lua_engine.dll", "dzclient_api.dll" };

        internal static void LoadYDWE()
        {
            string basePath = PluginSystem.HackPath;
            try
            {
                foreach (var dll in binDlls)
                    LoadLibraryA(Path.Combine(basePath, dll));

                foreach (var dll in pluginDlls)
                {
                    IntPtr BaseAddr = LoadLibraryA(Path.Combine(basePath, dll));
                    if (BaseAddr != IntPtr.Zero)
                    {
                        plugins.Add(dll, BaseAddr);
                        if (dll == "dzclient_api.dll")
                        {
                            CreateWindowHook.CreateWindowEvent = hWnd =>
                            {
                                IntPtr procAddr = Kernel32.GetProcAddress(BaseAddr, "SetWindow");
                                if (procAddr != IntPtr.Zero)
                                    Marshal.GetDelegateForFunctionPointer<SetWindowPrototype>(procAddr)(hWnd);
                            };
                        }
                    }
                }
            }
            catch
            {
                Trace.WriteLine("Library Load Failed!");
            }
        }

        internal static void Initialize()
        {
            try
            {
                string mixtapePath = Path.Combine(PluginSystem.HackPath, "Mixtape");
                if (Directory.Exists(mixtapePath))
                    foreach (var item in Directory.GetFiles(mixtapePath))
                        if (Path.GetExtension(item).Equals(".dll", StringComparison.OrdinalIgnoreCase))
                            LoadLibraryA(item);
                foreach (var plugin in plugins)
                {
                    if (plugin.Key != "dzclient_api.dll")
                    {
                        Trace.WriteLine($"Initializing {plugin.Key} . . .");
                        IntPtr procAddr = Kernel32.GetProcAddress(plugin.Value, "Initialize");
                        if (procAddr != IntPtr.Zero)
                            Marshal.GetDelegateForFunctionPointer<InitializePrototype>(procAddr)();
                    }
                    else if (PluginSystem.IsDebugging)
                    {
                        IntPtr ptr = plugin.Value + 0xB950;
                        if (Kernel32.VirtualProtect(ptr, 1, 0x40, out uint lpflOldProtect))
                        {
                            Marshal.WriteByte(ptr, 0x55);
                            Kernel32.VirtualProtect(ptr, 1, lpflOldProtect, out _);
                        }
                    }
                }
                Trace.WriteLine("Successed!");
            }
            catch
            {
                Trace.WriteLine("Initialize Failed!");
            }
        }

        internal static void StateReset()
        {
            if (plugins.ContainsKey("dzclient_api.dll"))
                Marshal.WriteInt32(plugins["dzclient_api.dll"] + 0x3365C, 0);
        }
    }
}
