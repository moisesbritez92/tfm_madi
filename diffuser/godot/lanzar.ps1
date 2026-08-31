# Lanzador de la demostracion Push-T en Godot.
#
# Arranca el servidor de politica y despues Godot, en la condicion que se pida,
# y se lleva por delante el servidor al terminar. Existe para no tener que
# recordar dos rutas largas, dos puertos y media docena de banderas delante de
# un tribunal.
#
#   .\lanzar.ps1                       condicion A, en vivo, semilla 10000
#   .\lanzar.ps1 -Obs godot            condicion B: la politica ve los pixeles de Godot
#   .\lanzar.ps1 -Semilla 200003       otra condicion inicial
#   .\lanzar.ps1 -Modo grabar          episodio completo a grabaciones\, sin ventana en A
#   .\lanzar.ps1 -Modo reproducir      reproduce lo grabado, sin GPU y sin servidor
#   .\lanzar.ps1 -Variante v3          pone DINOv2 congelada en el bucle en vez de V0
#   .\lanzar.ps1 -Obs godot -Perturbacion t_roja   la pieza se pinta de rojo
#
# Ninguna cifra que muestre es un resultado del TFM.

[CmdletBinding()]
param(
    [ValidateSet("estado", "godot")]
    [string]$Obs = "estado",

    [ValidateSet("v0", "v1", "v2", "v3", "v4")]
    [string]$Variante = "v0",

    # Solo tiene efecto en la condicion B: en la A la imagen la dibuja Python con
    # el codigo del entrenamiento, que no sabe nada de perturbaciones.
    [ValidateSet("ninguna", "t_roja", "sombras")]
    [string]$Perturbacion = "ninguna",

    [ValidateSet("vivo", "grabar", "reproducir", "observacion", "comparar", "cobertura")]
    [string]$Modo = "vivo",

    [int]$Semilla = 10000,
    [int]$Puerto = 5555,
    [double]$Velocidad = 1.0,

    # Ruta al ejecutable de Godot 4. Si no se pasa, se busca donde lo deja winget.
    [string]$Godot = ""
)

$ErrorActionPreference = "Stop"
$Proyecto = $PSScriptRoot
$Raiz = Split-Path (Split-Path $Proyecto -Parent) -Parent
$Python = Join-Path $Raiz ".venv_diffuser_infer\Scripts\python.exe"
$Servidor = Join-Path $Proyecto "servidor\servidor_politica.py"

function Resolver-Godot {
    if ($Godot -ne "") { return $Godot }
    $candidatos = @(
        "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\GodotEngine.GodotEngine_Microsoft.Winget.Source_8wekyb3d8bbwe\Godot_v4.7.2-stable_win64_console.exe",
        "$env:LOCALAPPDATA\Microsoft\WinGet\Links\godot_console.exe"
    )
    foreach ($c in $candidatos) { if (Test-Path $c) { return $c } }
    $winget = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Filter "Godot_v*_console.exe" -Recurse -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($winget) { return $winget.FullName }
    throw "no se encuentra Godot 4. Instalalo con 'winget install GodotEngine.GodotEngine' o pasa -Godot <ruta>"
}

$exe = Resolver-Godot
if (-not (Test-Path $Python)) { throw "falta el entorno de inferencia: $Python" }

# Los modos de verificacion no hablan con la politica, y reproducir tampoco: esa
# es justamente su gracia.
$necesitaServidor = $Modo -notin @("comparar", "cobertura", "reproducir")

$proceso = $null
if ($necesitaServidor) {
    Write-Host "arrancando el servidor de politica ($($Variante.ToUpper()), obs=$Obs, puerto $Puerto) ..." -ForegroundColor Cyan
    $proceso = Start-Process -FilePath $Python `
        -ArgumentList @($Servidor, "--variante", $Variante, "--obs", $Obs,
                        "--puerto", "$Puerto") `
        -PassThru -NoNewWindow

    # Cargar V0 son unos treinta segundos: se espera al puerto, no a un reloj.
    # Un TcpClient directo, y no Test-NetConnection, que resuelve nombres y hace
    # su propio sondeo antes de contestar.
    $listo = $false
    $limite = (Get-Date).AddSeconds(180)
    while ((Get-Date) -lt $limite) {
        if ($proceso.HasExited) { throw "el servidor de politica murio al arrancar" }
        try {
            $sonda = New-Object System.Net.Sockets.TcpClient
            $sonda.Connect("127.0.0.1", $Puerto)
            $sonda.Close()
            $listo = $true
            break
        } catch {
            Start-Sleep -Milliseconds 500
        }
    }
    if (-not $listo) { throw "el servidor no abrio el puerto $Puerto en 180 s" }
    Write-Host "servidor listo" -ForegroundColor Green
}

# La condicion B necesita renderizador: sin ventana no hay SubViewport que leer.
$sinVentana = ($Obs -eq "estado") -and ($Modo -in @("grabar", "comparar", "cobertura"))
$argumentos = @("--path", $Proyecto)
if ($sinVentana) { $argumentos = @("--headless") + $argumentos }
$argumentos += @("--", "modo=$Modo", "obs=$Obs", "seed=$Semilla", "puerto=$Puerto",
                 "velocidad=$Velocidad", "perturbacion=$Perturbacion", "variante=$Variante")

try {
    & $exe @argumentos
} finally {
    if ($proceso -and -not $proceso.HasExited) {
        Write-Host "parando el servidor de politica" -ForegroundColor Cyan
        Stop-Process -Id $proceso.Id -Force -ErrorAction SilentlyContinue
    }
}
