import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from itertools import combinations
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def df_to_markdown(df):
    headers = list(df.columns)
    sep = ["---"] * len(headers)
    rows = df.values.tolist()
    md  = "| " + " | ".join(headers) + " |\n"
    md += "| " + " | ".join(sep) + " |\n"
    for row in rows:
        md += "| " + " | ".join(str(v) for v in row) + " |\n"
    return md

def anomaly_clustering(filepath, k=4):
    df = pd.read_excel(filepath)
    df.columns = df.columns.str.strip()
    features = ['NETPRO', 'Q20Age', 'Q21Gender', 'Q22Income',
                'Q23FLY', 'Q5TIMESFLOWN', 'Q6LONGUSE']
    df_sub = df[features].dropna().copy()
    df_sub['Q21Gender'] = LabelEncoder().fit_transform(df_sub['Q21Gender'])
    df_sub['Q22Income'] = LabelEncoder().fit_transform(df_sub['Q22Income'])
    X = StandardScaler().fit_transform(df_sub)
    kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
    df_sub['cluster'] = kmeans.labels_
    perc = df_sub['cluster'].value_counts(normalize=True) * 100
    stats = []
    for c in sorted(df_sub['cluster'].unique()):
        pct = perc[c]
        prof = df_sub[df_sub['cluster']==c][features].mean()
        row = {'cluster': c, 'percent': round(pct,2)}
        row.update({f: round(prof[f],3) for f in features})
        stats.append(row)
    stats_df = pd.DataFrame(stats)
    outlier = stats_df.loc[stats_df['percent'].idxmin()]
    return stats_df, outlier

def association_rules(transactions, min_support=0.25, min_confidence=0.5):
    num = len(transactions)
    items = sorted({i for t in transactions for i in t})
    freq = {}
    for i in items:
        sup = sum(i in t for t in transactions) / num
        if sup >= min_support:
            freq[(i,)] = sup
    for combo in combinations(items, 2):
        sup = sum(set(combo).issubset(t) for t in transactions) / num
        if sup >= min_support:
            freq[combo] = sup
    rules = []
    for itemset, sup_item in freq.items():
        if len(itemset) < 2: continue
        for r in range(1, len(itemset)):
            for ant in combinations(itemset, r):
                cons = tuple(sorted(set(itemset) - set(ant)))
                sup_ant = freq.get(tuple(sorted(ant)), 0)
                if sup_ant and sup_item / sup_ant >= min_confidence:
                    rules.append({
                        'antecedent': ant,
                        'consequent': cons,
                        'support': round(sup_item,2),
                        'confidence': round(sup_item/sup_ant,2)
                    })
    return pd.DataFrame(rules)

def logistic_example():
    X, y = make_classification(n_samples=500, n_features=5,
                               n_informative=3, n_redundant=0,
                               n_classes=2, random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LogisticRegression().fit(X_tr, y_tr)
    report_dict = classification_report(y_te, model.predict(X_te), output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    return report_df, model.coef_[0], model.intercept_[0]

def generate_markdown(filepath, output_md="report.md"):
    stats_df, outlier = anomaly_clustering(filepath)
    transactions = [
        ['fever','cough'], ['cough','fatigue'], ['fever','fatigue'],
        ['fever','cough','fatigue'], ['headache','fever'],
        ['cough','headache'], ['fever','cough','headache'],
        ['cough','fatigue','headache']
    ]
    rules_df = association_rules(transactions)
    report_df, coefs, intercept = logistic_example()

    with open(output_md, 'w', encoding='utf-8') as f:
        f.write("## 1. Análise de Anomalia via Clustering\n\n")
        f.write(f"**Cluster atípico**: {int(outlier['cluster'])} — {outlier['percent']:.2f}% dos passageiros\n\n")
        f.write("**Perfil médio do cluster atípico:**\n\n")
        f.write(df_to_markdown(outlier.to_frame().T.drop(columns=['cluster'])) + "\n\n---\n\n")

        f.write("## 2. Regras de Associação (Sintomas de Pacientes)\n\n")
        f.write("**Transações:**\n```\n")
        for t in transactions:
            f.write(f"{t}\n")
        f.write("```\n\n")
        f.write("### Regras geradas:\n\n")
        if rules_df.empty:
            f.write("Nenhuma regra encontrada com os thresholds dados.\n\n")
        else:
            f.write(df_to_markdown(rules_df) + "\n\n")
        f.write("---\n\n")

        f.write("## 3. Regressão Logística (Exemplo Sintético)\n\n")
        f.write("### Relatório de Classificação\n\n")
        f.write(df_to_markdown(report_df) + "\n\n")
        f.write(f"**Coeficientes:** {coefs.tolist()}\n\n")
        f.write(f"**Intercept:** {intercept:.6f}\n")

    print(f"Markdown report gerado em: {output_md}")

if __name__ == "__main__":
    filepath = "./sfo_2018_data file_final_Weightedv2.xlsx"
    generate_markdown(filepath)
