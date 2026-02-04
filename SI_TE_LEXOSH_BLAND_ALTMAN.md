# Si të Lexosh Bland–Altman Plot (Measured vs Predicted)

## Përmbledhje

Bland–Altman plot tregon **pajtueshmërinë** midis vlerave të matura dhe atyre të parashikuara. Ai përgjigjet në pyetjen: **“Sa afër janë parashikimet me matjet reale?”**

Ky projekt krijon dy file:
- `results/bland_altman_b.png` (për fushën magnetike **B**, µT)
- `results/bland_altman_e.png` (për fushën elektrike **E**, V/m)

---

## Çfarë tregon një Bland–Altman plot?

### 1) Boshti X (Horizontal)
**Mesatarja e vlerës së matur dhe të parashikuar**:

```
(Measured + Predicted) / 2
```

Tregon “niveli mesatar” për secilin çift matje/parashikim.

### 2) Boshti Y (Vertikal)
**Diferenca** midis parashikimit dhe matjes:

```
Predicted – Measured
```

Nëse pika është:
- **mbi 0** → parashikimi është më i lartë se matja
- **poshtë 0** → parashikimi është më i ulët se matja

---

## Vijat kryesore në plot

### Vija e kuqe (Mean Difference)
Kjo është **mesatarja e diferencave**. Tregon nëse modeli ka **bias**:
- afër 0 → pa bias të dukshëm
- larg 0 → modeli priret të mbivlerësojë ose nënvlerësojë

### Dy vijat gri (Limits of Agreement)
Këto janë kufijtë:

```
Mean diff ± 1.96 * SD
```

Interpretim:
- Nëse shumica e pikave janë **brenda** këtyre vijave → pajtueshmëri e mirë
- Nëse shumë pika janë **jashtë** → modeli ka devijim të lartë

---

## Si ta lexosh një Bland–Altman plot (shpejt)

1. **Shiko vijen e kuqe**  
   - afër 0 → mirë  
   - larg 0 → bias

2. **Shiko sa pika janë brenda vijave gri**  
   - shumica brenda → stabilitet i mirë  
   - shumë jashtë → modeli nuk është konsistent

3. **Shiko formën e shpërndarjes**  
   - nëse diferencat rriten me rritjen e vlerës → modeli gabon më shumë në vlera të larta

---

## Shembull interpretimi

Nëse shihni:
- **Mean diff** = 0.2 µT  
- **LoA** = [-2.5, 2.9] µT  
- shumica e pikave brenda

atëherë:
✅ modeli është i pranueshëm  
❗ por gabimi mund të shkojë deri në ±2–3 µT

---

## Ku i gjen këto plot-e në projekt

### Bland–Altman standard
- `results/bland_altman_b.png` → për B (µT)  
- `results/bland_altman_e.png` → për E (V/m)

### Bland–Altman me ngjyra (sipas `car_type`)
- `results/bland_altman_b_by_car.png`
- `results/bland_altman_e_by_car.png`

### Legjenda e ngjyrave (car_type)
- electric → blu  
- hybrid → portokalli  
- mildhybrid → jeshile  
- unknown → gri

---

## Shënim i rëndësishëm

Bland–Altman është **plot diagnostik**. Ai nuk tregon sa “i saktë” është modeli në mënyrë absolute, por tregon **pajtueshmërinë** dhe **bias-in** midis matjes dhe parashikimit.

Në tezë, mund ta përdorësh për të përshkruar:
- nëse modeli është neutral (bias ≈ 0)
- sa i madh është devijimi maksimal i pritshëm