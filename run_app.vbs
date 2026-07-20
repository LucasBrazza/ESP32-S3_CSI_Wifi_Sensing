Set shell = CreateObject("WScript.Shell")
Set fileSystem = CreateObject("Scripting.FileSystemObject")

projectDir = fileSystem.GetParentFolderName(WScript.ScriptFullName)

shell.CurrentDirectory = projectDir
shell.Run Chr(34) & projectDir & "\run_app.bat" & Chr(34), 0, False