@echo off
set CSC="C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\Roslyn\csc.exe"
set JNDIR=%~dp0..\out
set OUT=%~dp0..\out\RLCommPlugin.dll

%CSC% /target:library /langversion:latest /out:%OUT% /reference:%JNDIR%\Cirnix.JassNative.Runtime.dll /reference:%JNDIR%\Cirnix.JassNative.dll /reference:%JNDIR%\Cirnix.JassNative.Common.dll /reference:%JNDIR%\EasyHook.dll /reference:%JNDIR%\Newtonsoft.Json.dll RLCommPlugin.cs
