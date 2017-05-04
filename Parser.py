from numpy import mean, std, insert

def _split_input(X, Y, percent_train, percent_valid):
    if percent_train + percent_valid >= 100:
        raise Exception

    input_size = len(X)
    limit_train = input_size * percent_train / 100
    limit_valid = input_size * (percent_train + percent_valid) / 100

    X_train = X[:limit_train]
    Y_train = Y[:limit_train]

    X_valid = X[limit_train:limit_valid]
    Y_valid = Y[limit_train:limit_valid]

    X_test = X[limit_valid:]
    Y_test = Y[limit_valid:]

    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test

def parse_ej1(percent_train, percent_valid, f_normalize=None):
    def validate_input(i):
        for x in i:
            if len(x) != 10:
                raise Exception("Una entrada falla la validacion")

    with open('tp1_ej1_training.csv', 'r') as f:
        input = f.read()
    lines = input.splitlines()
    lists = [line.split(',') for line in lines]
    parsed_values = [map(float, x[1:]) for x in lists]
    # el primero tiene 'M o 'B'
    parsed_expected = [[0] if x[0] == 'M' else [1] for x in lists]
    validate_input(parsed_values)
    if f_normalize is not None:
        f_normalize(parsed_values, parsed_expected)
    parsed_values = map(lambda x: insert(x, 0, -1), parsed_values)
    return _split_input(parsed_values, parsed_expected, percent_train, percent_valid)

def parse_ej2(percent_train, percent_valid, f_normalize=None):
    def validate_input(i):
        for x in i:
            if len(x) != 8:
                raise Exception("Una entrada falla la validacion")

    with open('tp1_ej2_training.csv', 'r') as f:
        input = f.read()
    lines = input.splitlines()
    lists = [line.split(',') for line in lines]

    #los ultimos dos son las respuestas
    parsed_values = [map(float, x[:-2]) for x in lists]
    parsed_expected = [[float(x[-1]),float(x[-2])] for x in lists]
    validate_input(parsed_values)
    if f_normalize is not None:
        parsed_values, parsed_expected = f_normalize(parsed_values, parsed_expected)
    parsed_values = map(lambda x: insert(x, 0, -1), parsed_values)


    return _split_input(parsed_values, parsed_expected, percent_train, percent_valid)

def normalize_minmax(X, Y):
    def normalize_list(V):
        # para cada parametro
        for param_num in range(0, len(V[0])):
            param_data = [pttrn[param_num] for pttrn in V]
            param_min = min(param_data)
            param_max = max(param_data)
            for pttrn in V:
                pttrn[param_num] = 2 * (pttrn[param_num] - param_min) / (param_max - param_min - 1)
    return normalize_list(X), normalize_list(Y)

def normalize_standarize(X, Y):
    def normalize_list(V):
        # para cada parametro
        for param_num in range(0, len(V[0])):
            param_data = [pttrn[param_num] for pttrn in V]
            param_mean = mean(param_data)
            param_std = std(param_data)
            for pttrn in V:
                pttrn[param_num] = (pttrn[param_num] - param_mean) / param_std
    return normalize_list(X), normalize_list(Y)
