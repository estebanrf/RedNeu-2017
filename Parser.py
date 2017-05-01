def parse_ej1():
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
    parsed_expected = [[0,1] if x[0] == 'M' else [1,0] for x in lists]
    validate_input(parsed_values)
    return parsed_values, parsed_expected

def parse_ej2():
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
    return parsed_values, parsed_expected
