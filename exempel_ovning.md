# Backpropagation - Övning

## Nätverksarkitektur

```
[3, 2, 3] = 3 inputs → 2 hidden neurons → 3 output neurons

        Input Layer          Hidden Layer         Output Layer

           x₀ ─────────────┬──── Neuron 1 ────┬──── Neuron 3 (y₀)
                           │                  │
           x₁ ─────────────┼──── Neuron 2 ────┼──── Neuron 4 (y₁)
                           │                  │
           x₂ ─────────────┘                  └──── Neuron 5 (y₂)
```

## Träningsexempel

| | Värde |
|---|---|
| **Input** | [0.0000, 0.4492, 0.1630] |
| **Target** | [0, 1, 0] |
| **Learning rate** | 0.2 |

---

# STEG 1: Forward Pass

> **Formel:** Output = sigmoid(Σ(Input × Vikt) + Bias)
>
> **Sigmoid:** σ(x) = 1 / (1 + e⁻ˣ)

## Lager 1 (Hidden)

### Neuron 1

| Input | Vikt | Input × Vikt |
|------:|-----:|-------------:|
| 0.0000 | 0.1394 | 0.0000 |
| 0.4492 | -0.4750 | ___ |
| 0.1630 | -0.2250 | -0.0367 |
| | **Summa:** | **-0.2500** |

| | Värde |
|---|---:|
| Bias | -0.2768 |
| Total (Summa + Bias) | -0.5268 |
| **Output** = sigmoid(-0.5268) | **0.3713** |

---

### Neuron 2

| Input | Vikt | Input × Vikt |
|------:|-----:|-------------:|
| 0.0000 | 0.2365 | 0.0000 |
| 0.4492 | 0.1767 | 0.0794 |
| 0.1630 | 0.3922 | 0.0639 |
| | **Summa:** | **0.1433** |

| | Värde |
|---|---:|
| Bias | -0.4131 |
| Total (Summa + Bias) | ___ |
| **Output** = sigmoid(___) | **___** |

---

## Lager 2 (Output)

### Neuron 3

| Input (från Hidden) | Vikt | Input × Vikt |
|--------------------:|-----:|-------------:|
| 0.3713 | -0.0781 | -0.0290 |
| 0.4330 | -0.4702 | -0.2036 |
| | **Summa:** | **-0.2326** |

| | Värde |
|---|---:|
| Bias | -0.2814 |
| Total (Summa + Bias) | -0.5139 |
| **Output** = sigmoid(-0.5139) | **0.3743** |

---

### Neuron 4

| Input (från Hidden) | Vikt | Input × Vikt |
|--------------------:|-----:|-------------:|
| 0.3713 | 0.0054 | 0.0020 |
| 0.4330 | -0.4735 | ___ |
| | **Summa:** | **___** |

| | Värde |
|---|---:|
| Bias | -0.3012 |
| Total (Summa + Bias) | -0.5042 |
| **Output** = sigmoid(-0.5042) | **0.3766** |

---

### Neuron 5

| Input (från Hidden) | Vikt | Input × Vikt |
|--------------------:|-----:|-------------:|
| 0.3713 | 0.1499 | 0.0556 |
| 0.4330 | 0.0449 | 0.0195 |
| | **Summa:** | **0.0751** |

| | Värde |
|---|---:|
| Bias | -0.2796 |
| Total (Summa + Bias) | -0.2045 |
| **Output** = sigmoid(-0.2045) | **0.4491** |

---

# STEG 2: Backward Pass (Felberäkning)

## Output-lagret (Lager 2)

> **Formel:** Delta = (Target - Output) × Output × (1 - Output)

### Neuron 3

| Steg | Formel | Värde |
|------|--------|------:|
| Target | | 0 |
| Output | | 0.3743 |
| Error | Target - Output | -0.3743 |
| Sigmoid derivata | Output × (1 - Output) | ___ |
| **Delta** | Error × Sigmoid derivata | **-0.0877** |

---

### Neuron 4

| Steg | Formel | Värde |
|------|--------|------:|
| Target | | 1 |
| Output | | 0.3766 |
| Error | Target - Output | ___ |
| Sigmoid derivata | Output × (1 - Output) | 0.2348 |
| **Delta** | Error × Sigmoid derivata | **___** |

---

### Neuron 5

| Steg | Formel | Värde |
|------|--------|------:|
| Target | | 0 |
| Output | | 0.4491 |
| Error | Target - Output | -0.4491 |
| Sigmoid derivata | Output × (1 - Output) | 0.2474 |
| **Delta** | Error × Sigmoid derivata | **-0.1111** |

---

## Hidden-lagret (Lager 1)

> **Formel:** Delta = (Σ Delta_next × Vikt_till_next) × Output × (1 - Output)

### Neuron 1

**Steg 1: Beräkna error contribution från output-lagret**

| Nästa Neuron | Delta | Vikt till Neuron 1 | Bidrag |
|--------------|------:|-------------------:|-------:|
| Neuron 3 | -0.0877 | -0.0781 | 0.0068 |
| Neuron 4 | 0.1464 | 0.0054 | 0.0008 |
| Neuron 5 | -0.1111 | 0.1499 | ___ |
| | | **Summa:** | **-0.0090** |

**Steg 2: Beräkna delta**

| Steg | Formel | Värde |
|------|--------|------:|
| Output | | 0.3713 |
| Error contribution | Summa från ovan | -0.0090 |
| Sigmoid derivata | Output × (1 - Output) | 0.2334 |
| **Delta** | Error contrib × Sigmoid derivata | **-0.0021** |

---

### Neuron 2

**Steg 1: Beräkna error contribution från output-lagret**

| Nästa Neuron | Delta | Vikt till Neuron 2 | Bidrag |
|--------------|------:|-------------------:|-------:|
| Neuron 3 | -0.0877 | -0.4702 | ___ |
| Neuron 4 | 0.1464 | -0.4735 | -0.0693 |
| Neuron 5 | -0.1111 | 0.0449 | -0.0050 |
| | | **Summa:** | **___** |

**Steg 2: Beräkna delta**

| Steg | Formel | Värde |
|------|--------|------:|
| Output | | 0.4330 |
| Error contribution | Summa från ovan | ___ |
| Sigmoid derivata | Output × (1 - Output) | 0.2455 |
| **Delta** | Error contrib × Sigmoid derivata | **-0.0081** |

---

# STEG 3: Viktuppdatering

> **Formel:** Ny vikt = Gammal vikt + Learning rate × Delta × Input
>
> **Learning rate = 0.2**

## Lager 1 (Hidden)

### Neuron 1 (δ = -0.0021)

| Vikt | Gammal | Input | Ändring | Ny |
|------|-------:|------:|--------:|---:|
| w₀ | 0.1394 | 0.0000 | 0.2 × -0.0021 × 0.0000 = -0.0000 | 0.1394 |
| w₁ | -0.4750 | 0.4492 | 0.2 × -0.0021 × 0.4492 = ___ | ___ |
| w₂ | -0.2250 | 0.1630 | 0.2 × -0.0021 × 0.1630 = -0.0001 | -0.2250 |
| bias | -0.2768 | 1 | 0.2 × -0.0021 = -0.0004 | -0.2772 |

---

### Neuron 2 (δ = -0.0081)

| Vikt | Gammal | Input | Ändring | Ny |
|------|-------:|------:|--------:|---:|
| w₀ | 0.2365 | 0.0000 | 0.2 × -0.0081 × 0.0000 = -0.0000 | 0.2365 |
| w₁ | 0.1767 | 0.4492 | 0.2 × -0.0081 × 0.4492 = -0.0007 | 0.1760 |
| w₂ | 0.3922 | 0.1630 | 0.2 × -0.0081 × 0.1630 = -0.0003 | 0.3919 |
| bias | -0.4131 | 1 | 0.2 × -0.0081 = -0.0016 | -0.4147 |

---

## Lager 2 (Output)

### Neuron 3 (δ = -0.0877)

| Vikt | Gammal | Input (från Hidden) | Ändring | Ny |
|------|-------:|--------------------:|--------:|---:|
| w₀ | -0.0781 | 0.3713 | 0.2 × -0.0877 × 0.3713 = -0.0065 | -0.0846 |
| w₁ | -0.4702 | 0.4330 | 0.2 × -0.0877 × 0.4330 = -0.0076 | -0.4778 |
| bias | -0.2814 | 1 | 0.2 × -0.0877 = -0.0175 | -0.2989 |

---

### Neuron 4 (δ = 0.1464)

| Vikt | Gammal | Input (från Hidden) | Ändring | Ny |
|------|-------:|--------------------:|--------:|---:|
| w₀ | 0.0054 | 0.3713 | 0.2 × 0.1464 × 0.3713 = ___ | ___ |
| w₁ | -0.4735 | 0.4330 | 0.2 × 0.1464 × 0.4330 = 0.0127 | -0.4608 |
| bias | -0.3012 | 1 | 0.2 × 0.1464 = 0.0293 | -0.2719 |

---

### Neuron 5 (δ = -0.1111)

| Vikt | Gammal | Input (från Hidden) | Ändring | Ny |
|------|-------:|--------------------:|--------:|---:|
| w₀ | 0.1499 | 0.3713 | 0.2 × -0.1111 × 0.3713 = -0.0082 | 0.1416 |
| w₁ | 0.0449 | 0.4330 | 0.2 × -0.1111 × 0.4330 = -0.0096 | 0.0353 |
| bias | -0.2796 | 1 | 0.2 × -0.1111 = -0.0222 | -0.3018 |

---

# Sammanfattning

| Neuron | Output (före) | Delta | Funktion |
|--------|---------------|-------|----------|
| 1 | 0.3713 | -0.0021 | Hidden |
| 2 | ___ | -0.0081 | Hidden |
| 3 | 0.3743 | -0.0877 | Output (y₀) |
| 4 | 0.3766 | ___ | Output (y₁) |
| 5 | 0.4491 | -0.1111 | Output (y₂) |

---
