## O Problema do Threshold Padrão

Todo modelo de classificação binária produz uma **probabilidade** entre 0 e 1 para cada observação. Para transformar essa probabilidade em uma classe predita (0 ou 1), é necessário definir um **threshold de corte**.

Por padrão, a maioria das bibliotecas usa threshold = 0.5. Mas esse valor é arbitrário — e quase nunca é o ideal para um problema real de negócio.

### Por que 0.5 raramente é o melhor threshold?

- Em datasets **desbalanceados** (ex: 5% de fraudes), o modelo tende a prever probabilidades baixas para a classe positiva. Um threshold de 0.5 perde quase todos os eventos
- Diferentes problemas têm **custos assimétricos**: detectar uma fraude perdida (falso negativo) pode custar muito mais do que um alarme falso (falso positivo)
- O threshold ótimo depende do **objetivo de negócio**, não de uma convenção estatística

### A pergunta que o threshold responde

```
A partir de qual probabilidade devo classificar uma observação como evento?

threshold = 0.3: classifica como positivo quem tem P(y=1) ≥ 0.3
  → captura mais eventos (maior Recall), mas aceita mais falsos positivos
  → menor Precision

threshold = 0.7: classifica como positivo quem tem P(y=1) ≥ 0.7
  → só classifica quando muito confiante (maior Precision)
  → perde mais eventos reais (menor Recall)

O threshold perfeito para o negócio está em algum ponto entre esses extremos.
```

## 2. Precision, Recall e a Matriz de Confusão

Antes de entender a curva, é fundamental dominar as duas métricas que ela combina.

### A Matriz de Confusão

```
                    Predito: Positivo   Predito: Negativo
Real: Positivo  |   TP (Verdadeiro +)  |  FN (Falso Negativo)  |
Real: Negativo  |   FP (Falso Positivo)|  TN (Verdadeiro -)    |

TP: evento real que o modelo acertou como evento
FN: evento real que o modelo classificou como não-evento  ← erro grave em fraude, diagnóstico
FP: não-evento que o modelo classificou como evento       ← alarme falso
TN: não-evento que o modelo acertou como não-evento
```

### Precision (Precisão)

```
Precision = TP / (TP + FP)

Pergunta: dos que o modelo disse serem positivos, quantos realmente são?

Precision alta → quando o modelo diz "evento", ele geralmente está certo
Precision baixa → muitos alarmes falsos

Exemplo: de 100 clientes classificados como churn, 70 realmente deram churn
Precision = 70/100 = 0.70
```

### Recall (Sensibilidade / TPR)

```
Recall = TP / (TP + FN)

Pergunta: dos eventos reais, quantos o modelo conseguiu capturar?

Recall alto → o modelo captura a maioria dos eventos reais
Recall baixo → o modelo perde muitos eventos reais

Exemplo: de 200 clientes que realmente deram churn, o modelo identificou 140
Recall = 140/200 = 0.70
```

### O Trade-off Fundamental

```
Aumentar threshold → Precision sobe, Recall cai
Diminuir threshold → Recall sobe, Precision cai

Esses dois movimentos são inevitáveis — melhorar um implica
piorar o outro. A curva Precision-Recall mapeia esse trade-off
para todos os thresholds possíveis.
```

## 3. A Curva Precision-Recall

A Curva Precision-Recall é construída variando o threshold de 0 a 1 e calculando Precision e Recall para cada valor. O resultado é uma curva que mapeia o trade-off completo — revelando qual threshold oferece o melhor equilíbrio para o objetivo de negócio.

### Como a curva é construída

```
Para threshold t = 0.0, 0.01, 0.02, ..., 1.0:
  1. Classificar: y_pred[i] = 1 se P(y=1|x_i) ≥ t, senão 0
  2. Calcular TP, FP, FN com base em y_pred e y_real
  3. Precision(t) = TP / (TP + FP)
  4. Recall(t)    = TP / (TP + FN)
  5. Plotar ponto (Recall(t), Precision(t))

Eixo X: Recall  (0 = não captura nada, 1 = captura tudo)
Eixo Y: Precision (0 = só alarmes falsos, 1 = sem alarmes falsos)

Curva ideal: sobe até (1, 1) — Precision=1 e Recall=1 simultaneamente
Curva basal: linha horizontal em Precision = taxa de eventos
             (o que um modelo aleatório produziria)
```

### Average Precision (AP) e AUC-PR

```
AUC-PR = área sob a curva Precision-Recall
       = Σ_t [Recall(t) - Recall(t-1)] × Precision(t)
       = média ponderada das precisões em cada threshold

Interpretação:
  AUC-PR = 1.0 → modelo perfeito (Precision=1 e Recall=1 simultaneamente)
  AUC-PR = taxa de eventos → modelo aleatório (baseline)

Por que AUC-PR é melhor que AUC-ROC para datasets desbalanceados?
  AUC-ROC usa TN no denominador do FPR → mascarado por TN abundantes
  AUC-PR não usa TN → foca apenas em como o modelo performa nos positivos
  → Em fraude (0.1% de positivos), AUC-ROC pode ser 0.98 enquanto
    o modelo praticamente não detecta nenhum evento
```

## 4. Implementação Completa com XGBoost

### 4.1. Treinar o modelo e gerar probabilidades

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split

# Treinar o modelo
model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='auc',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Probabilidades do conjunto de validação
y_proba = model.predict_proba(X_val)[:, 1]  # P(y=1 | X)

print(f"Taxa de eventos real:          {y_val.mean():.4f}")
print(f"Probabilidade média predita:   {y_proba.mean():.4f}")
print(f"Min prob: {y_proba.min():.4f} | Max prob: {y_proba.max():.4f}")
```

### 4.2. Calcular e plotar a Curva Precision-Recall

```python
# Calcular a curva
precisions, recalls, thresholds = precision_recall_curve(y_val, y_proba)
av_precision = average_precision_score(y_val, y_proba)
baseline = y_val.mean()  # precision de um modelo aleatório

print(f"Average Precision (AUC-PR): {av_precision:.4f}")
print(f"Baseline (taxa de eventos): {baseline:.4f}")
print(f"Ganho sobre baseline:       {av_precision/baseline:.2f}x")

# Plotar
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- Curva PR ---
ax1 = axes[0]
ax1.plot(recalls, precisions, color='steelblue', lw=2,
         label=f'XGBoost (AP = {av_precision:.3f})')
ax1.axhline(y=baseline, color='gray', linestyle='--', lw=1.5,
            label=f'Baseline aleatório ({baseline:.3f})')
ax1.set_xlabel('Recall', fontsize=12)
ax1.set_ylabel('Precision', fontsize=12)
ax1.set_title('Curva Precision-Recall', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.set_xlim([0, 1])
ax1.set_ylim([0, 1.05])
ax1.grid(alpha=0.3)

# --- Precision e Recall por threshold ---
ax2 = axes[1]
ax2.plot(thresholds, precisions[:-1], color='darkorange', lw=2, label='Precision')
ax2.plot(thresholds, recalls[:-1],    color='steelblue',  lw=2, label='Recall')
ax2.set_xlabel('Threshold', fontsize=12)
ax2.set_ylabel('Score', fontsize=12)
ax2.set_title('Precision e Recall por Threshold', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.set_xlim([0, 1])
ax2.set_ylim([0, 1.05])
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('precision_recall_curve.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 4.3. Encontrar o threshold ótimo

```python
# -------------------------------------------------------
# Critério 1: maximizar F1-Score
# (equilíbrio harmônico entre Precision e Recall)
# -------------------------------------------------------
f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (
    precisions[:-1] + recalls[:-1] + 1e-10
)
best_idx_f1 = np.argmax(f1_scores)
threshold_f1 = thresholds[best_idx_f1]

print("=== Threshold ótimo pelo F1-Score ===")
print(f"Threshold:  {threshold_f1:.4f}")
print(f"F1-Score:   {f1_scores[best_idx_f1]:.4f}")
print(f"Precision:  {precisions[best_idx_f1]:.4f}")
print(f"Recall:     {recalls[best_idx_f1]:.4f}")

# -------------------------------------------------------
# Critério 2: Recall mínimo garantido pelo negócio
# Ex: "quero capturar pelo menos 80% dos fraudes"
# -------------------------------------------------------
recall_minimo = 0.80
mask_recall = recalls[:-1] >= recall_minimo
if mask_recall.any():
    # Dentre os thresholds que atingem o recall mínimo,
    # escolher o que maximiza Precision
    precisions_filtradas = np.where(mask_recall, precisions[:-1], 0)
    best_idx_recall = np.argmax(precisions_filtradas)
    threshold_recall = thresholds[best_idx_recall]

    print(f"\n=== Threshold com Recall ≥ {recall_minimo} ===")
    print(f"Threshold:  {threshold_recall:.4f}")
    print(f"Precision:  {precisions[best_idx_recall]:.4f}")
    print(f"Recall:     {recalls[best_idx_recall]:.4f}")
    print(f"F1-Score:   {f1_scores[best_idx_recall]:.4f}")

# -------------------------------------------------------
# Critério 3: Precision mínima garantida pelo negócio
# Ex: "quero que pelo menos 70% dos alertas sejam verdadeiros"
# -------------------------------------------------------
precision_minima = 0.70
mask_prec = precisions[:-1] >= precision_minima
if mask_prec.any():
    recalls_filtrados = np.where(mask_prec, recalls[:-1], 0)
    best_idx_prec = np.argmax(recalls_filtrados)
    threshold_prec = thresholds[best_idx_prec]

    print(f"\n=== Threshold com Precision ≥ {precision_minima} ===")
    print(f"Threshold:  {threshold_prec:.4f}")
    print(f"Precision:  {precisions[best_idx_prec]:.4f}")
    print(f"Recall:     {recalls[best_idx_prec]:.4f}")
    print(f"F1-Score:   {f1_scores[best_idx_prec]:.4f}")
```

### 4.4. Avaliar o modelo com o threshold escolhido

```python
# Aplicar o threshold escolhido
threshold_final = threshold_f1  # ou threshold_recall / threshold_prec

y_pred_final = (y_proba >= threshold_final).astype(int)

# Relatório completo
print(f"Threshold aplicado: {threshold_final:.4f}")
print(f"\n{classification_report(y_val, y_pred_final, target_names=['Não-Evento','Evento'])}")

# Matriz de confusão detalhada
cm = confusion_matrix(y_val, y_pred_final)
tn, fp, fn, tp = cm.ravel()

print("Matriz de Confusão:")
print(f"  Verdadeiros Positivos (TP): {tp}")
print(f"  Falsos Positivos (FP):      {fp}  ← alarmes falsos")
print(f"  Falsos Negativos (FN):      {fn}  ← eventos perdidos")
print(f"  Verdadeiros Negativos (TN): {tn}")
print(f"\n  Precision:  {tp/(tp+fp):.4f}")
print(f"  Recall:     {tp/(tp+fn):.4f}")
print(f"  F1-Score:   {f1_score(y_val, y_pred_final):.4f}")
```

## 5. Análise de Custo-Benefício do Threshold

Nem sempre o F1-Score é a métrica certa para escolher o threshold. Em muitos problemas de negócio, os custos de Falso Positivo e Falso Negativo são radicalmente diferentes. A análise de custo-benefício encontra o threshold que **maximiza o valor econômico** da estratégia.

```python
# Exemplo: problema de churn
# - Recuperar um cliente vale R$ 500 (beneficio de TP)
# - Oferta para um cliente que não ia churnar custa R$ 50 (custo de FP)
# - Não detectar um churn custa R$ 200 (custo de FN)
# - TN: nenhum custo

benefit_tp  =  500   # valor de capturar um evento real
cost_fp     =  -50   # custo de alarme falso
cost_fn     = -200   # custo de evento perdido

# Calcular valor líquido para cada threshold
net_values = []
for i, thresh in enumerate(thresholds):
    y_pred_t = (y_proba >= thresh).astype(int)
    cm_t = confusion_matrix(y_val, y_pred_t)
    tn_t, fp_t, fn_t, tp_t = cm_t.ravel()
    net = tp_t * benefit_tp + fp_t * cost_fp + fn_t * cost_fn
    net_values.append(net)

best_cost_idx = np.argmax(net_values)
threshold_custo = thresholds[best_cost_idx]

print("=== Threshold ótimo pelo Custo-Benefício ===")
print(f"Threshold:       {threshold_custo:.4f}")
print(f"Valor líquido:   R$ {max(net_values):,.0f}")
print(f"Precision:       {precisions[best_cost_idx]:.4f}")
print(f"Recall:          {recalls[best_cost_idx]:.4f}")

# Comparar vs threshold 0.5 padrão
y_pred_default = (y_proba >= 0.5).astype(int)
cm_def = confusion_matrix(y_val, y_pred_default)
tn_d, fp_d, fn_d, tp_d = cm_def.ravel()
net_default = tp_d * benefit_tp + fp_d * cost_fp + fn_d * cost_fn

print(f"\nValor líquido com threshold=0.5: R$ {net_default:,.0f}")
print(f"Ganho do threshold otimizado:    R$ {max(net_values)-net_default:,.0f}")
```

## 6. Precision-Recall vs ROC — Quando usar cada uma?

| Critério | Curva ROC (AUC) | Curva Precision-Recall (AP) |
| --- | --- | --- |
| Foco | Discriminação geral | Performance nos positivos |
| Baseline | AUC = 0.5 (50%) | AP = taxa de eventos (pode ser 1%) |
| Datasets desbalanceados | Pode ser enganosa (TN inflam FPR) | Mais honesta e informativa |
| Datasets balanceados | Equivalente | Equivalente |
| Métrica de escolha do threshold | Não direta | Sim — Precision/Recall por threshold |
| Uso comum | Comparação geral de modelos | Fraude, diagnóstico, detecção de eventos raros |

### Regra prática

- Use **AUC-ROC** para comparar e ranquear modelos de forma geral
- Use **Curva Precision-Recall** para definir o threshold operacional, especialmente em datasets desbalanceados
- Use **análise de custo-benefício** quando os custos de FP e FN são conhecidos e assimétricos
- **Nunca use Accuracy** como métrica principal em datasets desbalanceados — um modelo que classifica tudo como negativo atinge 99% de accuracy se a taxa de eventos for 1%