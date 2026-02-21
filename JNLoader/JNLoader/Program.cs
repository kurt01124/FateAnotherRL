using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;

namespace JNLoader
{
	public static class Program
	{
		[DllImport("kernel32.dll")]
		private static extern IntPtr GetConsoleWindow();

		[DllImport("user32")]
		private static extern IntPtr FindWindow(string lpClassName, string lpWindowName);

		[DllImport("user32.dll")]
		private static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);

		public static int Main(string[] args)
		{
			StringBuilder stringBuilder = new StringBuilder();
			StringBuilder stringBuilder2 = new StringBuilder();
			string text = null;
			try
			{
				text = Path.GetDirectoryName(Assembly.GetEntryAssembly().Location);
				string text2 = text;
				string text3 = text;
				bool flag = false;
				for (int i = 0; i < args.Length; i++)
				{
					string text4 = args[i].ToLower();
					if (text4 == "-debug")
					{
						flag = true;
					}
					else if (text4 == "-skipupdate")
					{
						// No-op: JNUpdater auto-update removed
					}
					else if (text4 == "-path")
					{
						if (args.Length != i + 1)
						{
							string text6 = args[++i];
							if (text6[0] == '.')
							{
								text2 = text + "\\" + text6;
							}
							else
							{
								text2 = text6;
							}
						}
					}
					else if (text4 == "-libpath")
					{
						if (args.Length != i + 1)
						{
							string text5 = args[++i];
							if (text5[0] == '.')
							{
								text3 = text + "\\" + text5;
							}
							else
							{
								text3 = text5;
							}
						}
					}
					else if (text4 == "-loadfile")
					{
						stringBuilder.Append(args[i] + " ");
					}
					else
					{
						if (args[i][0] == '-')
						{
							stringBuilder.Append(args[i] + " ");
						}
						else
						{
							stringBuilder.Append("\"" + args[i] + "\" ");
						}
					}
					if (args[i][0] == '-')
					{
						stringBuilder2.Append(args[i] + " ");
					}
					else
					{
						stringBuilder2.Append("\"" + args[i] + "\" ");
					}
				}
				foreach (Process process in Process.GetProcessesByName("war3"))
				{
					try
					{
						if (process.MainWindowHandle == IntPtr.Zero)
						{
							process.Kill();
						}
						process.Dispose();
					}
					catch
					{
					}
				}
				foreach (Process process2 in Process.GetProcessesByName("Warcraft III"))
				{
					try
					{
						if (process2.MainWindowHandle == IntPtr.Zero)
						{
							process2.Kill();
						}
						process2.Dispose();
					}
					catch
					{
					}
				}
				byte[] bytes = Encoding.UTF8.GetBytes(text2);
				for (int k = 0; k < bytes.Length; k++)
				{
					if (bytes[k] >= 128)
					{
						Console.WriteLine("[JNLoader] WARNING: Path contains non-ASCII characters, may cause issues.");
						break;
					}
				}
				try
				{
					File.WriteAllText(text3 + "\\logs\\args.txt", stringBuilder2.ToString().Trim());
				}
				catch
				{
				}
				int num = Launcher.Start(text2, text3, stringBuilder.ToString().Trim(), flag);
				switch (num)
				{
				case -3:
					Console.WriteLine("[JNLoader] ERROR: Warcraft III version mismatch (need 1.28.5.7680)");
					return 0;
				case -2:
					Console.WriteLine("[JNLoader] ERROR: Unsupported parameter");
					return 0;
				case -1:
					Console.WriteLine("[JNLoader] ERROR: war3.exe not found in: " + text2);
					return 0;
				default:
					return num;
				}
			}
			catch (FileNotFoundException ex)
			{
				Console.WriteLine("[JNLoader] ERROR: File not found: " + ex.FileName);
			}
			catch (Exception ex2)
			{
				Console.WriteLine("[JNLoader] ERROR: " + ex2.Message);
			}
			return 0;
		}
	}
}
