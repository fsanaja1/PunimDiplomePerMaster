# Zgjidhja e Problemit: Python nuk Gjetet

## Problemi

Kur ekzekutoni `python .\src\rf_elf_model.py`, merrni mesazhin:
```
Python was not found; run without arguments to install from the Microsoft Store...
```

## Shkaku

Windows ka një "App Execution Alias" për Python që çon në Microsoft Store në vend që të ekzekutojë Python të instaluar. Kjo ndodh kur Python nuk është instaluar ose nuk është në PATH.

## Zgjidhja 1: Instaloni Python (Rekomanduar)

### Hapi 1: Shkarkoni Python
1. Shkoni në: https://www.python.org/downloads/
2. Shkarkoni versionin më të fundit të Python 3.x (rekomandohet Python 3.11 ose më i ri)

### Hapi 2: Instaloni Python
1. Ekzekutoni installer-in e shkarkuar
2. **E RËNDËSISHME**: Zgjidhni "Add Python to PATH" në fillim të instalimit
3. Zgjidhni "Install Now" ose "Customize installation"
4. Nëse zgjidhni "Customize", sigurohuni që "Add Python to environment variables" është e zgjedhur

### Hapi 3: Verifikoni Instalimin
Hapni një PowerShell të ri dhe ekzekutoni:
```powershell
python --version
```

Duhet të shfaqet diçka si: `Python 3.11.x`

## Zgjidhja 2: Çaktivizoni Windows App Execution Aliases

Nëse keni Python të instaluar tashmë por nuk funksionon:

1. Hapni **Settings** në Windows
2. Shkoni te **Apps** → **Advanced app settings** → **App execution aliases**
3. Gjeni "python.exe" dhe "python3.exe"
4. **Çaktivizoni** të dyja (kthejeni toggle në OFF)
5. Mbyllni dhe rihapni PowerShell

## Zgjidhja 3: Përdorni Python Launcher (py)

Nëse keni Python të instaluar por `python` nuk funksionon, provoni:

```powershell
py .\src\rf_elf_model.py
```

Ose:

```powershell
py -3 .\src\rf_elf_model.py
```

## Zgjidhja 4: Përdorni Rrugë të Plotë

Nëse e dini ku është instaluar Python, përdorni rrugën e plotë:

```powershell
# Shembull (ndryshoni rrugën sipas instalimit tuaj):
C:\Python311\python.exe .\src\rf_elf_model.py
```

## Verifikimi i Instalimit

Pas instalimit, verifikoni që gjithçka funksionon:

```powershell
# Kontrolloni versionin
python --version

# Kontrolloni që pip funksionon
pip --version

# Kontrolloni që jeni në direktoriun e duhur
cd C:\Users\florin.sanaja\Documents\punimdiplome\PunimDiplomePerMaster

# Instaloni varësitë
python -m pip install -r requirements.txt

# Ekzekutoni skriptin
python .\src\rf_elf_model.py
```

## Nëse Problemi Vazhdon

1. **Rihapni PowerShell** pas instalimit (ose çaktivizimit të aliases)
2. **Verifikoni PATH**: 
   ```powershell
   $env:PATH -split ';' | Select-String -Pattern 'Python'
   ```
3. **Shtoni Python në PATH manualisht** nëse nuk është aty:
   - Hapni "Environment Variables" në Windows
   - Shtoni rrugën e Python në System PATH

## Rekomandim

**Zgjidhja më e mirë** është të instaloni Python nga python.org dhe të siguroheni që "Add Python to PATH" është e zgjedhur gjatë instalimit.


