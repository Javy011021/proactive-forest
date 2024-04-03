
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler


def assign_number(word, data):
    if word not in data:
        data[word] = len(data)
    return data[word]

def normalize(column):
    scaler = MinMaxScaler()
    column= scaler.fit_transform(column.to_frame())
    return column

def string_to_number(X):
    for i in range(len(X.columns)):
        column = X.iloc[:, i]
        if column.dtype == 'object':
            additional_object = {}
            column = column.apply(lambda x: assign_number(x, additional_object))
            column = normalize(column)
            X.iloc[:, i] = column
                    

def probabilites_chi2(X, y):
    
    string_to_number(X)

    selector = SelectKBest(score_func=chi2, k=2)
    selector.fit_transform(X, y)
    selector.get_support()

    total_score = sum(selector.scores_)
    normalized_scores = [score / total_score for score in selector.scores_]

    if(sum(normalized_scores)!=1):
        normalized_scores = [round(score, 10) for score in normalized_scores]
        
    return normalized_scores


    