def loop_on_indicies(shape):
    nb_dim = len(shape)
    size = 1
    for elt in shape:
        size *= elt
    index = [0] * nb_dim
    # Indicateur pour savoir quand arrêter de parcourir l'array
    done = False
    if size == 0:
        done = True
    # Tant que nous n'avons pas fini de parcourir l'array
    while not done:
        yield tuple(index)
        index[-1] += 1
        for i in reversed(range(nb_dim)):
            if index[i] >= shape[i]:
                index[i] = 0
                # Si nous sommes à la dernière dimension, nous avons fini de parcourir l'array
                if i == 0:
                    done = True

                # Sinon, nous passons à la dimension suivante
                else:
                    index[i - 1] += 1