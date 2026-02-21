using System;
using System.Diagnostics;
using System.IO;
using System.Threading;
using EasyHook;
using JNLoader.FileManage;

namespace JNLoader
{
	// Token: 0x02000002 RID: 2
	internal static class Launcher
	{
		// Token: 0x06000001 RID: 1 RVA: 0x00002050 File Offset: 0x00000250
		private static int CheckVersion(string path)
		{
			if (!File.Exists(path))
			{
				return -1;
			}
			if (!(FileVersionInfo.GetVersionInfo(path).FileVersion == "1.28.5.7680"))
			{
				return -3;
			}
			return 0;
		}

		// Token: 0x06000002 RID: 2 RVA: 0x00002078 File Offset: 0x00000278
		internal static int Start(string InstallPath, string JNServicePath, string InCommandLine, bool DebugMode)
		{
			string text = InstallPath + "\\war3.exe";
			string text2 = JNServicePath + "\\Cirnix.JassNative.Runtime.dll";
			int num = Launcher.CheckVersion(text);
			if (num < 0)
			{
				int num2 = Launcher.CheckVersion(text = InstallPath + "\\Warcraft III.exe");
				if (num2 < 0)
				{
					if (num >= num2)
					{
						return num2;
					}
					return num;
				}
			}
			DirectoryInfo directoryInfo = new DirectoryInfo(JNServicePath + "\\logs");
			if (directoryInfo.Exists)
			{
				foreach (FileInfo fileInfo in directoryInfo.GetFiles("*", SearchOption.AllDirectories))
				{
					if (fileInfo.CreationTime.AddMonths(1) < DateTime.Now)
					{
						try
						{
							fileInfo.Delete();
						}
						catch
						{
						}
					}
				}
			}
			// Check environment variable for injection mode
			string injectMode = Environment.GetEnvironmentVariable("JNLOADER_INJECT_MODE");

			int i;
			try
			{
				if (injectMode == "proxy")
				{
					// Proxy DLL mode: ijl15.dll proxy bootstraps JassNative (Wine/Docker compatible)
					// No EasyHook injection - war3.exe loads ijl15.dll naturally via PE imports
					Console.WriteLine("[JNLoader] Proxy DLL mode (Wine/Docker compat)");

					string originalIjl = Path.Combine(InstallPath, "ijl15.dll");
					string backupIjl = Path.Combine(InstallPath, "ijl15_original.dll");
					string proxyIjl = Path.Combine(InstallPath, "ijl15_proxy.dll");

					if (!File.Exists(proxyIjl))
					{
						Console.WriteLine("[JNLoader] ERROR: ijl15_proxy.dll not found at " + proxyIjl);
						return -10;
					}

					// Backup original ijl15.dll if not already done
					if (!File.Exists(backupIjl))
					{
						if (File.Exists(originalIjl))
						{
							Console.WriteLine("[JNLoader] Backing up: ijl15.dll -> ijl15_original.dll");
							File.Copy(originalIjl, backupIjl, false);
						}
					}

					// Install proxy as ijl15.dll
					Console.WriteLine("[JNLoader] Installing proxy: ijl15_proxy.dll -> ijl15.dll");
					File.Copy(proxyIjl, originalIjl, true);

					// Start war3.exe directly (no injection)
					var psi = new ProcessStartInfo
					{
						FileName = text,
						Arguments = InCommandLine,
						WorkingDirectory = InstallPath,
						UseShellExecute = false
					};
					Console.WriteLine("[JNLoader] Starting war3.exe (proxy DLL will bootstrap JassNative)");
					var proc = Process.Start(psi);
					Console.WriteLine("[JNLoader] war3.exe PID=" + proc.Id);
					proc.WaitForExit();
					Console.WriteLine("[JNLoader] war3.exe exited with code: " + proc.ExitCode);
					i = proc.Id;
				}
				else if (injectMode == "delayed")
				{
					// Wine compat mode: start process then inject
					Console.WriteLine("[JNLoader] Delayed inject mode (Wine compat)");
					var psi = new ProcessStartInfo
					{
						FileName = text,
						Arguments = InCommandLine,
						WorkingDirectory = InstallPath,
						UseShellExecute = false
					};
					var proc = Process.Start(psi);
					int delaySec = 5;
					string delayEnv = Environment.GetEnvironmentVariable("JNLOADER_DELAY");
					if (delayEnv != null) int.TryParse(delayEnv, out delaySec);
					Console.WriteLine("[JNLoader] war3.exe started, PID=" + proc.Id + ", waiting " + delaySec + "s...");
					Thread.Sleep(delaySec * 1000);
					Console.WriteLine("[JNLoader] Injecting (NoService, NoWOW64Bypass)...");
					try
					{
						RemoteHooking.Inject(proc.Id,
							InjectionOptions.NoService | InjectionOptions.NoWOW64Bypass,
							text2, text2, DebugMode, JNServicePath, InstallPath);
						Console.WriteLine("[JNLoader] Inject OK");
					}
					catch (Exception ex)
					{
						Console.WriteLine("[JNLoader] Inject error (may be OK on Wine): " + ex.Message);
						Console.WriteLine("[JNLoader] Continuing - JassNative may have loaded successfully");
					}
					Console.WriteLine("[JNLoader] Waiting for war3.exe to exit...");
					proc.WaitForExit();
					Console.WriteLine("[JNLoader] war3.exe exited with code: " + proc.ExitCode);
					i = proc.Id;
				}
				else
				{
					// Default mode: CreateAndInject (Windows native)
					int num3;
					RemoteHooking.CreateAndInject(text, InCommandLine, 0, text2, text2, out num3, new object[] { DebugMode, JNServicePath, InstallPath });
					i = num3;
				}
			}
			catch (ArgumentException)
			{
				i = -2;
			}
			return i;
		}
	}
}
