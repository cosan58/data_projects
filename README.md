# Comparação de Classificadores — Credit Data

Projeto de estudo comparando três algoritmos de Machine Learning aplicados a um dataset de crédito, com **30 iterações** de treino/teste para validação estatística dos resultados.

## Algoritmos avaliados

| Algoritmo | Biblioteca |
|---|---|
| Naive Bayes (Gaussiano) | `scikit-learn` |
| Regressão Logística | `scikit-learn` |
| Random Forest | `scikit-learn` |

## Métricas reportadas

Para cada modelo, são exibidas:
- Precisão média
- Moda
- Mediana
- Variância
- Desvio padrão
- Coeficiente de variação

## Estrutura do projeto

\`\`\`
.
├── data/
│   └── credit_data.csv   # Dataset (não versionado — veja .gitignore)
├── main.py               # Script principal
├── requirements.txt      # Dependências
├── .gitignore
└── README.md
\`\`\`

## Como usar

### 1. Clone o repositório
\`\`\`bash
git clone https://github.com/cosan58/data_projects.git
cd data_projects
\`\`\`

### 2. Instale as dependências
\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 3. Execute
\`\`\`bash
python main.py
\`\`\`

## Licença
MIT
