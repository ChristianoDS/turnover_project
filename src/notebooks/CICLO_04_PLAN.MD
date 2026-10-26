# Ciclo 04 — Plano de Execução

Problema: classificação binária de turnover (`Attrition`: `Yes`/`No` → `1`/`0`).

O Ciclo 04 é o **ciclo de produtização**. Saímos dos notebooks exploratórios
(ciclos 1–3) para um **pacote Python em classes**, com uma `main` que treina o
modelo final e o aplica no conjunto de **teste** (lacrado desde o Ciclo 01).

**Modelo final:** Logistic Regression (venceu no treino/validação do Ciclo 03),
com os melhores hiperparâmetros encontrados no fine-tuning.

> Decisão metodológica do ciclo: o modelo final é re-treinado em **treino +
> validação combinados** (a validação já cumpriu o papel de ajudar nas decisões
> dos ciclos anteriores; agora vira dado de treino). O **teste** permanece
> intocado e é avaliado **uma única vez**.

---

## 1. Objetivos do ciclo

1. Modularizar o pipeline em **4 classes** (SRP, padrão `fit`/`transform`).
2. Criar uma `main.py` que treina em treino+val e avalia no teste.
3. Serializar o pipeline treinado com **joblib** (artefato para produção).
4. Criar um notebook que **importa** as classes e demonstra o fluxo ponta a
   ponta, aplicando o modelo no teste.
5. Fechar o ciclo treino → produção: o `app.py` Streamlit e o `predict.py`
   passam a carregar o pipeline serializado.

## 2. Estrutura de arquivos (estende `src/turnover`)

```
src/turnover/
├── __init__.py
├── config.py                # constantes: TARGET, caminhos, seed, best_params
├── schema.py                # (existe) contrato de colunas
├── data.py                  # (existe) load_csv + validate
├── feature_engineering.py   # classe FeatureEngineer (fit/transform)
├── preprocessing.py         # classe Preprocessor (RobustScaler + OneHotEncoder)
├── model.py                 # classe ModelWrapper (LogisticRegression)
├── pipeline.py              # classe TurnoverPipeline (orquestra tudo)
├── evaluate.py              # ml_metrics + escolha de threshold
└── predict.py               # (existe) passa a carregar o pipeline salvo

artifacts/                   # turnover_pipeline.joblib (gerado pelo train.py)
train.py                     # entrypoint de TREINO: treina + avalia no teste + salva
main.py                      # (existe) entrypoint de SCORING de um CSV novo
tests/                       # testes unitários das classes
```

> Nota: o `main.py` que já existe é o CLI de **scoring** (aplica o modelo num
> CSV). O **treino** fica em `train.py` separado, para não misturar os dois
> papéis. Após o `train.py` gerar o artefato, o `predict.py`/`main.py` passam a
> carregá-lo.

## 3. As 4 classes

Todas seguem o padrão `fit`/`transform` do sklearn — aprendem **só no treino** e
reaplicam sem recalcular, o que elimina leakage por construção.

### 3.1 `FeatureEngineer` (feature_engineering.py)

- `fit(df)`: aprende estatísticas de grupo do treino (ex.: mediana de renda por
  `JobLevel`).
- `transform(df)`: cria as features do Ciclo 02 — razões/taxas linha a linha
  (seguras) + as de grupo (usando o que foi aprendido no `fit`).
- `fit_transform(df)`.

### 3.2 `Preprocessor` (preprocessing.py)

- `fit(X)`: identifica colunas numéricas/categóricas, ajusta `RobustScaler` e
  `OneHotEncoder`, guarda a ordem final das colunas (`feature_names_`).
- `transform(X)`: aplica scaler + encoder e **reindexa pelas colunas do treino**
  (garante que teste tenha exatamente as mesmas colunas, na mesma ordem).

### 3.3 `ModelWrapper` (model.py)

- Encapsula a `LogisticRegression` com os `best_params` do Ciclo 03 e
  `class_weight='balanced'`.
- `fit(X, y)`, `predict_proba(X)`, `predict(X, threshold)`.
- `save(path)` / `load(path)` via joblib.

### 3.4 `TurnoverPipeline` (pipeline.py) — classe central

- Orquestra `FeatureEngineer → Preprocessor → ModelWrapper` como uma unidade.
- `fit(df_train)`: encadeia o fit das três etapas.
- `predict_proba(df)` / `predict(df, threshold)`: encadeia o transform + modelo.
- `save(path)` / `load(path)`: serializa o pipeline **inteiro** (as três etapas
  já fitadas) com joblib.

## 4. Módulos de apoio

- **config.py** — fonte única de constantes: `TARGET_COLUMN` (reusa schema),
  `SEED`, `BEST_PARAMS` (da LR tunada no Ciclo 03), `BEST_THRESHOLD` (da seção 11
  do Ciclo 03), caminhos de dados e artefatos.
- **evaluate.py** — `ml_metrics(nome, y_true, y_prob, threshold)` (migrado do
  notebook) e função de escolha do melhor threshold via curva precision-recall.

## 5. `train.py` — fluxo de execução (entrypoint de treino)

```
load_csv → validate → split (treino / val / teste)
   │
   ├── treino + val  → COMBINA → pipeline.fit()   [LR com best_params]
   │                                   │
   │                                   └── pipeline.save(artifacts/…)
   └── teste         → pipeline.predict_proba() → evaluate() [UMA vez]
```

Passos:
1. Carrega e valida o CSV (reusa `data.load_csv` + `data.validate`).
2. Split treino/teste e treino/val (mesmos `random_state` dos ciclos anteriores).
3. **Combina treino + validação** para o treino final.
4. `pipeline.fit(treino+val)`.
5. `pipeline.predict_proba(teste)` + `evaluate` no `BEST_THRESHOLD`.
6. `pipeline.save(artifacts/turnover_pipeline.joblib)`.

## 6. Notebook `ciclo_04.ipynb`

- Não re-implementa nada: **importa** `from turnover.pipeline import TurnoverPipeline`
  e as demais classes.
- Demonstra: carga/validação → split → combina treino+val → `fit` → aplica no
  teste → métricas → salva o artefato.
- Serve como documentação executável do uso do pacote.

## 7. Integração com produção

- `predict.py` (já existe, hoje é placeholder) passa a:
  `pipeline = TurnoverPipeline.load(ARTIFACT_PATH); return pipeline.predict_proba(df)`,
  mantendo a assinatura `score(df) -> df` que o `app.py` Streamlit já consome.

---

## Pontos de Atenção / Segurança

- **Desserialização (joblib/pickle):** carregar `.joblib` executa código na
  desserialização. O artefato **só** pode vir de fonte confiável (gerado pela
  própria `main`), nunca de upload de usuário. O caminho do artefato é fixo no
  `config.py`, não recebido de input externo.
- **Caminho de dados:** `DATA_PATH` resolvido a partir da raiz do projeto, não
  de input do usuário, evitando path traversal.
- **Sem leakage:** `fit` só em treino+val; teste intocado até a avaliação final.
- **Validação de schema:** `data.validate` roda antes de qualquer processamento
  (contrato de colunas em `schema.py`).
- **Testes:** cada classe com teste unitário; incluir teste de que o pipeline
  recusa DataFrame sem as colunas obrigatórias.
- **Determinismo:** `SEED` fixo em splits e modelo, para reprodutibilidade.
