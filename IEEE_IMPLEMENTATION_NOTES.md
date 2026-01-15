# Implementimi i Krahasimit me IEEE C95.6-2002

## Përmbledhje

Kodi është përditësuar për të përfshirë krahasimin me standardin **IEEE C95.6-2002** përveç ICNIRP. Kjo plotëson një nga kriteret kryesore që mungonte në tezë.

## Ndryshimet e Bëra

### 1. **Konstantat IEEE** (rreshtat ~183-210)
- Shtuar `IEEE_FREQUENCY_HZ = 50`
- Shtuar `IEEE_CATEGORY = "general_public"` (ose "occupational")
- Shtuar `IEEE_LIMIT_B_UT_BY_CATEGORY` me vlerat:
  - General public: **904 µT**
  - Occupational: **2710 µT**

### 2. **Llogaritja e Statistikave IEEE** (në funksionin `_train_and_plot_for_sheet`)
- Shtuar `exceeds_ieee` - kontrollon nëse vlera maksimale tejkalon kufirin IEEE
- Shtuar `exceed_margin_ieee` - llogarit diferencën nga kufiri IEEE

### 3. **Vizualizimet** (në plot-et PNG)
- Plot-et tani shfaqin informacion për të dyja standardet:
  - ICNIRP limit dhe status (exceeds: YES/NO)
  - IEEE C95.6-2002 limit dhe status (exceeds: YES/NO)

### 4. **Tabelat e Rezultateve**
- **`exposure_risk_by_sheet.csv`** - Tabelë e re që përfshin të dyja standardet
- **`icnirp_risk_by_sheet.csv`** - Ruhet për përputhshmëri me versionin e vjetër
- Të dyja tabelat përmbajnë kolona për:
  - `exceeds_icnirp`, `exceed_margin_icnirp`
  - `exceeds_ieee`, `exceed_margin_ieee`

### 5. **Summary File** (`model_results.txt`)
- Shtuar seksion për IEEE C95.6-2002
- Përditësuar "Per-sheet measured risk" për të shfaqur të dyja standardet

## ⚠️ VERIFIKIMI I VLERAVE IEEE

**E RËNDËSISHME**: Vlerat e IEEE C95.6-2002 që janë përdorur në kod duhet të **verifikohen** me dokumentin zyrtar të IEEE C95.6-2002.

### Vlerat Aktuale në Kod:
```python
IEEE_LIMIT_B_UT_BY_CATEGORY = {
    "general_public": 904.0,   # µT për 50 Hz
    "occupational": 2710.0,    # µT për 50 Hz
}
```

### Si të Verifikoni:
1. **Shkarkoni standardin IEEE C95.6-2002** nga:
   - https://standards.ieee.org/standard/C95_6-2002.html
   - Ose nga biblioteka akademike

2. **Gjeni tabelën për 50 Hz** në dokumentin IEEE

3. **Përditësoni vlerat në kod** nëse janë të ndryshme:
   - Ndryshoni vlerat në `IEEE_LIMIT_B_UT_BY_CATEGORY` (rreshtat ~200-203)
   - Ose ndryshoni `IEEE_CATEGORY` nëse përdorni një kategori tjetër

### Shënim për Vlerat IEEE:
IEEE C95.6-2002 përdor terminologji të ndryshme dhe mund të ketë interpretime të ndryshme:
- **Controlled environment** vs **Uncontrolled environment**
- Disa burime citojnë vlera të ndryshme për 50 Hz
- **Verifikoni gjithmonë me dokumentin zyrtar** që do të citoni në tezë

## Struktura e Rezultateve

### CSV File: `exposure_risk_by_sheet.csv`
Kolona të reja:
- `ieee_limit_b_ut` - Kufiri IEEE për këtë kategori
- `exceeds_ieee` - Boolean: TRUE nëse tejkalon kufirin IEEE
- `exceed_margin_ieee` - Diferenca nga kufiri IEEE (µT)

### Plot-et PNG
Tani shfaqin në fund:
```
Battery: 100% | What-if B̂: X.XXX µT | Measured max(B): X.XXX µT | 
ICNIRP(50 Hz, general_public): 200 µT (Exceeds: YES/NO) | 
IEEE C95.6-2002(50 Hz, general_public): 904 µT (Exceeds: YES/NO)
```

## Testimi

Pas përditësimit, ekzekutoni:
```powershell
python .\src\rf_elf_model.py
```

Kontrolloni:
1. ✅ Plot-et PNG përmbajnë informacion IEEE
2. ✅ `exposure_risk_by_sheet.csv` përmban kolona IEEE
3. ✅ `model_results.txt` përmban seksion IEEE
4. ✅ Të gjitha skenarët krahasohen me të dyja standardet

## Përfitimet për Tezën

1. ✅ **Plotëson kriterin e IEEE** - Tani kodi krahasohet me të dyja standardet
2. ✅ **Analizë më e plotë** - Mund të diskutoni dallimet midis ICNIRP dhe IEEE
3. ✅ **Rezultate më të besueshme** - IEEE zakonisht ka kufij më të lartë, kështu që më pak skenarë do të tejkalojnë
4. ✅ **Kontribut shkencor** - Krahasimi i dy standardeve është vlerësues për literaturën

## Hapat e Ardhshëm

1. **Verifikoni vlerat IEEE** me dokumentin zyrtar
2. **Përditësoni vlerat** nëse është e nevojshme
3. **Ekzekutoni analizën** për të gjitha të dhënat
4. **Diskutoni rezultatet** në tezë - pse disa skenarë tejkalojnë ICNIRP por jo IEEE (ose anasjelltas)




