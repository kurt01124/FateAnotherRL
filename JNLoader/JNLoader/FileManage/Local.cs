using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;

namespace JNLoader.FileManage
{
	// Token: 0x02000004 RID: 4
	internal static class Local
	{
		// Token: 0x06000007 RID: 7 RVA: 0x00002694 File Offset: 0x00000894
		internal static void ForceInstall(string Path, byte[] bytes)
		{
			try
			{
				if (Local.CheckDelete(Path))
				{
					File.WriteAllBytes(Path, bytes);
				}
			}
			catch
			{
			}
		}

		// Token: 0x06000008 RID: 8 RVA: 0x000026C8 File Offset: 0x000008C8
		internal static void CheckDirectory(string Path)
		{
			if (!Directory.Exists(Path))
			{
				Directory.CreateDirectory(Path);
			}
		}

		// Token: 0x06000009 RID: 9 RVA: 0x000026DC File Offset: 0x000008DC
		internal static bool CheckInstall(string Path, byte[] bytes)
		{
			bool flag;
			try
			{
				if (!File.Exists(Path))
				{
					File.WriteAllBytes(Path, bytes);
				}
				else
				{
					using (SHA256CryptoServiceProvider sha256CryptoServiceProvider = new SHA256CryptoServiceProvider())
					{
						IEnumerable<byte> enumerable = sha256CryptoServiceProvider.ComputeHash(bytes);
						byte[] array = sha256CryptoServiceProvider.ComputeHash(File.ReadAllBytes(Path));
						if (enumerable.SequenceEqual(array))
						{
							return false;
						}
						File.WriteAllBytes(Path, bytes);
					}
				}
				flag = true;
			}
			catch
			{
				flag = false;
			}
			return flag;
		}

		// Token: 0x0600000A RID: 10 RVA: 0x0000275C File Offset: 0x0000095C
		internal static bool CheckDelete(string Path)
		{
			bool flag;
			try
			{
				if (File.Exists(Path))
				{
					File.Delete(Path);
				}
				flag = true;
			}
			catch
			{
				flag = false;
			}
			return flag;
		}
	}
}
