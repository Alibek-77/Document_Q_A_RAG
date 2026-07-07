def tokenizing_document(doc:list[str]):
    tokens=[]
    for d in doc:
        for token in d.lower().split():
            tokens.append(token)
    return tokens
print(tokenizing_document(["ALibek krash sila","lkfs fdksf fdsk lf"]))