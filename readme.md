## 1. Análise de Anomalia via Clustering

**Cluster atípico**: 2 — 5.70% dos passageiros

**Perfil médio do cluster atípico:**

| percent | NETPRO | Q20Age | Q21Gender | Q22Income | Q23FLY | Q5TIMESFLOWN | Q6LONGUSE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5.7 | 9.938 | 0.575 | 0.212 | 0.075 | 0.094 | 2.394 | 2.5 |


---

## 2. Regras de Associação (Sintomas de Pacientes)

**Transações:**
```
['fever', 'cough']
['cough', 'fatigue']
['fever', 'fatigue']
['fever', 'cough', 'fatigue']
['headache', 'fever']
['cough', 'headache']
['fever', 'cough', 'headache']
['cough', 'fatigue', 'headache']
```

### Regras geradas:

| antecedent | consequent | support | confidence |
| --- | --- | --- | --- |
| ('cough',) | ('fatigue',) | 0.38 | 0.5 |
| ('fatigue',) | ('cough',) | 0.38 | 0.75 |
| ('cough',) | ('fever',) | 0.38 | 0.5 |
| ('fever',) | ('cough',) | 0.38 | 0.6 |
| ('cough',) | ('headache',) | 0.38 | 0.5 |
| ('headache',) | ('cough',) | 0.38 | 0.75 |
| ('fatigue',) | ('fever',) | 0.25 | 0.5 |
| ('headache',) | ('fever',) | 0.25 | 0.5 |


---

## 3. Regressão Logística (Exemplo Sintético)

### Relatório de Classificação

| precision | recall | f1-score | support |
| --- | --- | --- | --- |
| 0.8589743589743589 | 0.9305555555555556 | 0.8933333333333333 | 72.0 |
| 0.9305555555555556 | 0.8589743589743589 | 0.8933333333333333 | 78.0 |
| 0.8933333333333333 | 0.8933333333333333 | 0.8933333333333333 | 0.8933333333333333 |
| 0.8947649572649572 | 0.8947649572649572 | 0.8933333333333333 | 150.0 |
| 0.8961965811965812 | 0.8933333333333333 | 0.8933333333333333 | 150.0 |


**Coeficientes:** [-0.225433896242216, 2.0786885163408093, -0.14424662907390678, 0.33336548283151357, 0.4374524231382316]

**Intercept:** 0.125530
