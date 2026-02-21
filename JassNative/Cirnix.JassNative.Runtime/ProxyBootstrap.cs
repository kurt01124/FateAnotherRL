using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;

namespace Cirnix.JassNative.Runtime
{
    /// <summary>
    /// Entry point for proxy DLL mode (Wine/headless compatible).
    /// Called via CLR hosting: ExecuteInDefaultAppDomain.
    ///
    /// This class intentionally does NOT reference any EasyHook types
    /// so it can be JIT-compiled before the assembly resolver is registered.
    /// The actual EntryPoint usage is deferred to DoInit() which is compiled
    /// only after the resolver is in place.
    /// </summary>
    public class ProxyBootstrap
    {
        /// <summary>
        /// CLR hosting entry point. Signature must be: static int Method(string arg).
        /// arg = hackPath (JNService folder path)
        /// </summary>
        public static int ProxyInit(string hackPath)
        {
            try
            {
                // Flat layout: hackPath = installPath = WC3 root
                string installPath = hackPath;

                // Register assembly resolver BEFORE any EasyHook types are accessed.
                // This allows the CLR to find EasyHook.dll when JIT-compiling DoInit().
                AppDomain.CurrentDomain.AssemblyResolve += (object sender, ResolveEventArgs args) =>
                {
                    string file = string.Empty;
                    if (args.Name.IndexOf(',') >= 0)
                        file = args.Name.Substring(0, args.Name.IndexOf(',')) + ".dll";
                    else if (args.Name.IndexOf(".dll") >= 0)
                        file = Path.GetFileName(args.Name);
                    else
                        return null;

                    // Search paths: hackPath (JNService), installPath (WC3 root), and subdirectories
                    string[] searchDirs = new string[]
                    {
                        hackPath,
                        Path.Combine(hackPath, "plugins"),
                        Path.Combine(hackPath, "lib"),
                        installPath
                    };

                    foreach (string dir in searchDirs)
                    {
                        if (!Directory.Exists(dir)) continue;
                        try
                        {
                            string found = Directory.GetFiles(dir, file, SearchOption.AllDirectories).FirstOrDefault();
                            if (!string.IsNullOrEmpty(found))
                            {
                                try
                                {
                                    return Assembly.LoadFrom(found);
                                }
                                catch
                                {
                                    return Assembly.Load(File.ReadAllBytes(found));
                                }
                            }
                        }
                        catch { }
                    }
                    return null;
                };

                // Set environment flag so EntryPoint.Run knows we're in proxy mode
                Environment.SetEnvironmentVariable("JNLOADER_PROXY_MODE", "1");

                // Check if we should skip JassNative for testing
                string skipInit = Environment.GetEnvironmentVariable("JNLOADER_SKIP_INIT");
                if (skipInit == "1")
                {
                    Console.Error.WriteLine("[ProxyBootstrap] SKIP_INIT mode - CLR hosted but JassNative not loaded");
                    System.Threading.Thread.Sleep(System.Threading.Timeout.Infinite);
                    return 0;
                }

                // Call DoInit in a separate method to defer EasyHook type resolution
                return DoInit(hackPath, installPath);
            }
            catch (Exception ex)
            {
                try
                {
                    string logDir = Path.Combine(hackPath, "logs", "runtime");
                    if (!Directory.Exists(logDir))
                        Directory.CreateDirectory(logDir);
                    File.WriteAllText(
                        Path.Combine(logDir, $"proxy_error_{DateTime.Now:yyyy-MM-dd_HH.mm.ss}.log"),
                        ex.ToString());
                }
                catch { }

                // Write to stderr for Docker logs
                Console.Error.WriteLine("[ProxyBootstrap] Fatal: " + ex);
                return -1;
            }
        }

        /// <summary>
        /// Deferred initialization that references EntryPoint (which requires EasyHook).
        /// NoInlining prevents the JIT from pulling EntryPoint references into ProxyInit.
        /// </summary>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static int DoInit(string hackPath, string installPath)
        {
            // Create EntryPoint instance (constructor body is empty, null IContext is safe)
            var entry = new EntryPoint(null, false, hackPath, installPath);

            // Run blocks forever (Thread.Sleep(Infinite)) to keep hooks alive
            entry.Run(null, false, hackPath, installPath);

            return 0; // Never reached
        }
    }
}
