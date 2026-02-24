import numpy as np
import pandas as pd
import statistics as sts
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# --------------------------
# Implementação de Regressão Logística, Naive Bayes e Random Forest
# com 30 iterações para validação de resultados
# --------------------------

def load_data(filepath: str) -> tuple:
    """Carrega e pré-processa o dataset de crédito."""
    df = pd.read_csv(filepath)
    df.dropna(inplace=True)
    X = df.iloc[:, 1:4].values  # Features
    y = df.iloc[:, 4].values    # Target
    return X, y


def display_scores(scores: np.ndarray, nome_modelo: str) -> None:
    """Exibe medidas de posição e dispersão dos resultados dos testes."""
    print(f'Medidas de posição e dispersão — {nome_modelo}')
    print('-' * 50)
    print(f'Precisão média:          {scores.mean():.4f}')
    print(f'Moda:                    {sts.mode(scores)}')
    print(f'Mediana:                 {sts.median(scores):.4f}')
    print(f'Variância:               {sts.variance(scores):.6f}')
    print(f'Desvio padrão:           {sts.stdev(scores):.6f}')
    print(f'Coeficiente de variação: {stats.variation(scores) * 100:.4f}%')
    print()


def run_experiments(X: np.ndarray, y: np.ndarray, n_iter: int = 30) -> dict:
    """Executa n_iter experimentos para cada modelo e retorna os resultados."""
    resultados = {
        'Naive Bayes': [],
        'Regressão Logística': [],
        'Random Forest': [],
    }

    for i in range(n_iter):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            stratify=y,
            random_state=i
        )

        # Naive Bayes
        nb = GaussianNB()
        nb.fit(X_train, y_train)
        resultados['Naive Bayes'].append(accuracy_score(y_test, nb.predict(X_test)))

        # Regressão Logística
        lr = LogisticRegression()
        lr.fit(X_train, y_train)
        resultados['Regressão Logística'].append(accuracy_score(y_test, lr.predict(X_test)))

        # Random Forest
        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)
        resultados['Random Forest'].append(accuracy_score(y_test, rf.predict(X_test)))

    return {k: np.array(v) for k, v in resultados.items()}


def main():
    # Ajuste o caminho do arquivo conforme seu ambiente
    FILEPATH = 'data/credit_data.csv'

    X, y = load_data(FILEPATH)
    resultados = run_experiments(X, y, n_iter=30)

    for nome, scores in resultados.items():
        display_scores(scores, nome)


if __name__ == '__main__':
    main()
