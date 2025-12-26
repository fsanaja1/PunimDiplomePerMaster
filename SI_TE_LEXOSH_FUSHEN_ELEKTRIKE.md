# Si të Lexosh Informacionin për Fushën Elektrike (E) në Plot-et PNG

## Përmbledhje

Në plot-et e kombinuara (`*_combined.png`), informacioni për fushën elektrike shfaqet në **subplot-in e tretë (djathtas)**. Ky dokument shpjegon se si të interpretohet kjo informacion.

---

## Struktura e Plot-it të Kombinuar

Çdo plot i kombinuar përmban **3 subplot-e**:

1. **Majtas**: Feature Importance (mesatarja)
2. **Qendër**: Shpërndarja e B (µT) - Fusha Magnetike
3. **Djathtas**: Shpërndarja e E (V/m) - **Fusha Elektrike** ⚡

---

## Subplot-i i Fushës Elektrike (E)

### Çfarë tregon?

Subplot-i i tretë (djathtas) tregon **histogramin e shpërndarjes** së vlerave të fushës elektrike për të gjitha skenarët (sheets) e një lloji makine.

### Elementet e Plot-it:

#### 1. **Histogrami (Bars)**
- **Çfarë është**: Shpërndarja e vlerave të matura të fushës elektrike
- **Njësia**: Volt për metër (V/m)
- **Ngjyra**: Blu (#4A90E2)
- **Si të lexohet**:
  - Lartësia e çdo bar tregon **sa matje** kanë vlera në atë interval
  - Gjerësia e bar-ave tregon intervalin e vlerave
  - Më shumë bar-e të larta = më shumë matje me vlera në atë interval

#### 2. **Boshti X (Horizontal)**
- **Etiketa**: "E (V/m)"
- **Çfarë tregon**: Vlerat e fushës elektrike në Volt për metër
- **Shembull**: Nëse bar-i është në pozicionin 1.5, kjo do të thotë që ka matje me vlera rreth 1.5 V/m

#### 3. **Boshti Y (Vertical)**
- **Etiketa**: "Frequency"
- **Çfarë tregon**: Numri i matjeve që kanë vlera në atë interval
- **Shembull**: Nëse Y = 10, kjo do të thotë që 10 matje kanë vlera në atë interval

#### 4. **Titulli**
- **Format**: "Distribution of E (V/m) – {car_type}"
- **Shembull**: "Distribution of E (V/m) – electric"
- **Çfarë tregon**: Ky është shpërndarja e fushës elektrike për makinat elektrike

---

## Si të Interpretohet Shpërndarja

### 1. **Pika Më të Larta (Peaks)**
- **Çfarë do të thotë**: Intervalet ku ka më shumë matje
- **Shembull**: Nëse ka një peak në 1.2 V/m, kjo do të thotë që shumica e matjeve kanë vlera rreth 1.2 V/m

### 2. **Gjerësia e Shpërndarjes**
- **Shpërndarje e ngushtë**: Vlerat janë të afërta me njëra-tjetrën (më pak variacion)
- **Shpërndarje e gjerë**: Vlerat janë të shpërndara më shumë (më shumë variacion)

### 3. **Vlerat Maksimale dhe Minimale**
- **Maksimumi**: Vlera më e lartë e matur (në anën e djathtë të histogramit)
- **Minimumi**: Vlera më e ulët e matur (në anën e majtë të histogramit)

---

## Statistikat në Fund të Plot-it

Në fund të plot-it (në kutinë e tekstit), gjeni statistikat e kombinuara:

### Format:
```
Combined stats: B: mean=X.XXX µT, max=X.XXX µT | E: mean=X.XXX V/m, max=X.XXX V/m | ...
```

### Për Fushën Elektrike (E):

#### **E: mean=X.XXX V/m**
- **Çfarë është**: Mesatarja e të gjitha vlerave të matura të fushës elektrike
- **Si të lexohet**: Vlera mesatare e ekspozimit ndaj fushës elektrike
- **Shembull**: `E: mean=1.234 V/m` do të thotë që mesatarisht, ekspozimi është 1.234 V/m

#### **E: max=X.XXX V/m**
- **Çfarë është**: Vlera maksimale e matur e fushës elektrike
- **Si të lexohet**: Vlera më e lartë e ekspozimit që u mat
- **Shembull**: `E: max=2.570 V/m` do të thotë që vlera më e lartë e matur është 2.570 V/m

---

## Krahasimi me Standardet

**Shënim i rëndësishëm**: Në plot-et aktuale, **nuk ka vija referimi për kufijtë e ICNIRP/IEEE për fushën elektrike** (E), sepse fokusi kryesor është në fushën magnetike (B).

Megjithatë, nëse dëshironi të shtoni kufijtë për E, mund të përdorni:
- **ICNIRP për E**: ~5000 V/m (për publikun e përgjithshëm në 50 Hz)
- **IEEE për E**: Vlera të ndryshme në varësi të standardit

---

## Shembull i Leximit

Le të themi që shikoni plot-in `feature_importance_electric_combined.png`:

### Në Subplot-in e Fushës Elektrike (djathtas):

1. **Histogrami**: Shfaq shpërndarjen e vlerave
2. **Vlerat më të shpeshta**: Nëse ka një peak në 1.2 V/m, kjo do të thotë që shumica e matjeve kanë vlera rreth 1.2 V/m
3. **Gama e vlerave**: Nëse histogrami shkon nga 0.5 V/m deri në 2.0 V/m, kjo tregon që të gjitha matjet janë në këtë interval

### Në Tekstin në Fund:

```
E: mean=1.234 V/m, max=2.570 V/m
```

- **mean=1.234 V/m**: Mesatarisht, ekspozimi është 1.234 V/m
- **max=2.570 V/m**: Vlera më e lartë e matur është 2.570 V/m

---

## Pse Është e Rëndësishme?

1. **Vlerësimi i Ekspozimit**: Tregon se sa të larta janë vlerat e fushës elektrike në mjedise të ndryshme
2. **Krahasimi midis Makinave**: Mund të krahasoni shpërndarjet midis electric, hybrid, dhe mildhybrid
3. **Identifikimi i Skenarëve të Rrezikshëm**: Skenarët me vlera më të larta tregohen në histogram

---

## Tips për Interpretim

1. **Krahasoni me B**: Shikoni si ndryshon shpërndarja e E në krahasim me B
2. **Krahasoni midis car types**: Shikoni plot-et për electric, hybrid, dhe mildhybrid
3. **Përdorni CSV files**: CSV files përmbajnë të dhëna më të detajuara për analizë më të thellë

---

## Ku të Gjeni Të Dhënat e Detajuara

### CSV Files:
- `feature_importance_{car_type}_combined.csv` - Përmban feature importances
- `exposure_risk_by_sheet.csv` - Përmban statistikat për B dhe E për çdo sheet

### Në CSV, gjeni kolona:
- `measured_mean_e` - Mesatarja e E për atë sheet
- `measured_max_e` - Maksimumi i E për atë sheet
- `what_if_pred_e` - Parashikimi i E për what-if scenario

---

## Konkluzion

Subplot-i i fushës elektrike tregon **shpërndarjen e vlerave të matura** të fushës elektrike për të gjitha skenarët e një lloji makine. Duke e lexuar saktë, mund të:
- Kuptoni nivelin mesatar të ekspozimit
- Identifikoni vlerat maksimale
- Krahasoni midis llojeve të ndryshme të makinave
- Vlerësoni rrezikun potencial

