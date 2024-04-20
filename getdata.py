import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

scaler = StandardScaler()
enc = OneHotEncoder(sparse_output=False)

train_data = pd.read_csv("./data/train.csv")
train_ids = train_data["Id"]
y_train = train_data["SalePrice"]
train_data.drop(columns = ["Alley", "PoolQC", "MiscFeature", "Fireplaces", "MasVnrType",
                           "FireplaceQu", "Fence", "Id"], inplace=True)
objects = train_data.select_dtypes('object')
dummies = enc.fit_transform(objects)
dummies_names = enc.get_feature_names_out(objects.columns)
#print(objects)
#train_data = pd.get_dummies(train_data, dummy_na=True, dtype=int)
numbers = train_data.select_dtypes(exclude='object')
numbers = numbers.drop(columns=["SalePrice"])
number_names = numbers.columns
numbers_scaled = pd.DataFrame(scaler.fit_transform(numbers), columns = number_names)
train_data = pd.concat([numbers_scaled,
                        pd.DataFrame(dummies, columns=dummies_names).astype(int),
                        train_data["SalePrice"]], axis=1)
cor = train_data.corr()
mask = cor["SalePrice"].loc[lambda x: (x < 0.05) & (x > -0.05)].index
train_data.drop(mask, axis=1, inplace=True)
for column in ["LotFrontage", "MasVnrArea", "GarageYrBlt"]:
    train_data[column] = train_data[column].fillna(train_data[column].mean())



def get_data(test_ratio):
    #X_train = pd.concat([numbers_scaled,
    #                     pd.DataFrame(dummies, columns=dummies_names).astype(int)], axis=1)
    X_train = train_data.drop("SalePrice", axis=1)
    y_train = train_data["SalePrice"]
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_ratio)
    print(f"X_train: {X_train.iloc[0]}")
    print(f"X_val : {X_val.iloc[0]}")

    test_data = pd.read_csv("./data/test.csv")
    test_ids = test_data["Id"]
    test_data.drop(columns = ["Alley", "PoolQC", "MiscFeature", "Fireplaces", "MasVnrType",
                             "FireplaceQu", "Fence", "Id"], inplace=True)
    test_objects = test_data.select_dtypes("object")
    test_dummies = enc.transform(objects)
    
    test_numericals = test_data.select_dtypes(exclude="object")
    test_numerical_column_names = test_numericals.columns
    test_numericals_scaled = pd.DataFrame(scaler.transform(test_numericals), columns = number_names)
    
    
    test_data = pd.concat([test_numericals_scaled,
                           pd.DataFrame(test_dummies[0:-1], columns = dummies_names).astype(int)], axis=1)
    
    test_data.drop(mask, axis=1, inplace=True)
    nulls = test_data.isnull().sum().loc[lambda x: x > 0]
    for column in nulls.index:
        test_data[column] = test_data[column].fillna(test_data[column].mode()[0])
    
    return X_train, y_train, X_val, y_val, test_ids, test_data
