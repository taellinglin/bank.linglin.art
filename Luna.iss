; LunaCoin Inno Setup Script
; Basic version - ensure this file exists in the same folder as build.bat

[Setup]
AppName=Luna Suite
AppVersion=1.0.0
AppVerName=Luna Suite 1.0.0
AppPublisher=Your Company
AppPublisherURL=https://yourwebsite.com/
DefaultDirName={commonpf}\Luna Suite
DefaultGroupName=Luna Suite
OutputDir=Output
OutputBaseFilename=Luna_Setup
Compression=lzma
SolidCompression=yes
PrivilegesRequired=admin
WizardStyle=modern

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopnode"; Description: "Create LunaNode desktop icon"; GroupDescription: "Desktop Icons:"; Flags: unchecked
Name: "desktopwallet"; Description: "Create LunaWallet desktop icon"; GroupDescription: "Desktop Icons:"; Flags: unchecked

[Files]
Source: "dist\LunaNode.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "dist\LunaWallet.exe"; DestDir: "{app}"; Flags: ignoreversion

; Optional: Include default config files if they exist
; Source: "default_configs\blockchain.json"; DestDir: "{commonappdata}\Luna Suite"; Flags: onlyifdoesntexist uninsneveruninstall
; Source: "default_configs\mempool.json"; DestDir: "{commonappdata}\Luna Suite"; Flags: onlyifdoesntexist uninsneveruninstall
; Source: "default_configs\known_peers.json"; DestDir: "{commonappdata}\Luna Suite"; Flags: onlyifdoesntexist uninsneveruninstall

[Icons]
Name: "{group}\LunaNode"; Filename: "{app}\LunaNode.exe"
Name: "{group}\LunaWallet"; Filename: "{app}\LunaWallet.exe"
Name: "{group}\Uninstall Luna Suite"; Filename: "{uninstallexe}"
Name: "{autodesktop}\LunaNode"; Filename: "{app}\LunaNode.exe"; Tasks: desktopnode
Name: "{autodesktop}\LunaWallet"; Filename: "{app}\LunaWallet.exe"; Tasks: desktopwallet

[Run]
Filename: "{app}\LunaNode.exe"; Description: "Run LunaNode"; Flags: nowait postinstall skipifsilent
Filename: "{app}\LunaWallet.exe"; Description: "Run LunaWallet"; Flags: nowait postinstall skipifsilent unchecked

[Dirs]
Name: "{commonappdata}\Luna Suite"; Permissions: users-modify

[Code]
procedure InitializeWizard();
begin
  // Optional: Custom initialization code
end;