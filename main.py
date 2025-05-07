import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from itertools import combinations
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def anomaly_clustering(filepath, k=4):
    
    df = pd.read_excel(filepath)
    df.columns = df.columns.str.strip()
    
    features = ['NETPRO', 'Q20Age', 'Q21Gender', 'Q22Income',
                'Q23FLY', 'Q5TIMESFLOWN', 'Q6LONGUSE']
    df_sub = df[features].dropna().copy()
    
    df_sub['Q21Gender'] = LabelEncoder().fit_transform(df_sub['Q21Gender'])
    df_sub['Q22Income'] = LabelEncoder().fit_transform(df_sub['Q22Income'])
    
    X = StandardScaler().fit_transform(df_sub)
    
    kmeans = KMeans(n_clusters=k, random_state=42)
    df_sub['cluster'] = kmeans.fit_predict(X)
    
    perc = df_sub['cluster'].value_counts(normalize=True) * 100
    stats = []
    for c in sorted(df_sub['cluster'].unique()):
        pct = perc[c]
        profile = df_sub[df_sub['cluster']==c][features].mean().to_dict()
        stats.append({
            'cluster': c,
            'percent': pct,
            **{f"{f}_mean": profile[f] for f in features}
        })
    stats_df = pd.DataFrame(stats)
    
    outlier = stats_df.loc[stats_df['percent'].idxmin()]
    print("=== Estatísticas de Clusters ===")
    print(stats_df.to_string(index=False))
    print("\nCluster atípico:")
    print(outlier.to_string())
    return stats_df, outlier


def association_rules(transactions, min_support=0.25, min_confidence=0.5):
    num_trans = len(transactions)
    items = sorted({i for t in transactions for i in t})
    

    freq = {}

    for i in items:
        sup = sum(1 for t in transactions if i in t) / num_trans
        if sup >= min_support:
            freq[(i,)] = sup

    for combo in combinations(items, 2):
        sup = sum(1 for t in transactions if set(combo).issubset(t)) / num_trans
        if sup >= min_support:
            freq[combo] = sup
    
    rules = []
    for itemset, sup_itemset in freq.items():
        if len(itemset) < 2:
            continue
        for r in range(1, len(itemset)):
            for antecedent in combinations(itemset, r):
                consequent = tuple(sorted(set(itemset) - set(antecedent)))
                sup_ant = freq.get(tuple(sorted(antecedent)), 0)
                if sup_ant > 0:
                    conf = sup_itemset / sup_ant
                    if conf >= min_confidence:
                        rules.append({
                            'antecedent': antecedent,
                            'consequent': consequent,
                            'support': round(sup_itemset, 2),
                            'confidence': round(conf, 2)
                        })
    rules_df = pd.DataFrame(rules)
    print("\n=== Regras de Associação ===")
    if rules_df.empty:
        print("Nenhuma regra encontrada com os thresholds dados.")
    else:
        print(rules_df.to_string(index=False))
    return rules_df

# 3. REGRESSÃO LOGÍSTICA (exemplo sintético)
def logistic_example():
    # Gera dados
    X, y = make_classification(n_samples=500, n_features=5,
                               n_informative=3, n_redundant=0,
                               n_classes=2, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    coef = model.coef_[0]
    intercept = model.intercept_[0]
    
    print("\n=== Relatório de Classificação (Logistic Regression) ===")
    print(report)
    print(f"Coeficientes: {coef}")
    print(f"Intercept: {intercept}")
    return model, report

if __name__ == "__main__":
    filepath = "./sfo_2018_data file_final_Weightedv2.xlsx"
    
    stats_df, outlier = anomaly_clustering(filepath, k=4)
    
    transactions = [
        ['fever','cough'],
        ['cough','fatigue'],
        ['fever','fatigue'],
        ['fever','cough','fatigue'],
        ['headache','fever'],
        ['cough','headache'],
        ['fever','cough','headache'],
        ['cough','fatigue','headache']
    ]
    rules_df = association_rules(transactions,
                                 min_support=0.25,
                                 min_confidence=0.5)
    
    model, report = logistic_example()
