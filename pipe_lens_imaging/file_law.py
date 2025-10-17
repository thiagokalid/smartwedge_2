import numpy as np

__all__ = ['read', 'write']

# Índices de cada informação no .law:
idx_numR = 0
idx_numS = 1
idx_numT = 2
idx_numL = 3
idx_numV = 4
idx_retE = 5
idx_ampE = 6
idx_retR = 7
idx_ampR = 8


def write(root, emission, reception=None, elem_range=None, delay_time_unit="us"):
    if root[-4:] != ".law":
        root += ".law"

    header = [
        "# LOIS DE RETARD \n",
        "Version 1.0 \n",
        "numR\t"
        "numS\t"
        "numT\t"
        "numL\t"
        "numV\t"
        "retE\t"
        "ampE\t"
        "retR\t"
        "ampR\n"
    ]

    if elem_range is None:
        elem_range = [0, emission.shape[1] - 1]

    emission = np.copy(emission)
    if reception is None:
        reception = np.zeros_like(emission)
    else:
        reception = np.copy(reception)

    match delay_time_unit:
        case "s":
            emission *= 1e6
            reception *= 1e6
        case "ms":
            emission *= 1e3
            reception *= 1e3
        case "us":
            emission *= 1
            reception *= 1
        case _:
            raise ValueError("Invalid time unit")


    with open(root, "w") as file:
        file.writelines(header)

        for shot in range(0, emission.shape[0]):
            for elem_idx in range(elem_range[0], elem_range[-1] + 1):
                numR = 0  #
                numS = 0  #
                numT = shot  # Shot
                numL = 0  #
                numV = elem_idx + 1  # Índice do Emissor
                retE = emission[shot, elem_idx]  # Lei focal na Emissão
                ampE = 1  # Ganho na Emissão
                retR = reception[shot, elem_idx]  # Lei focal na Recepção
                ampR = 1  # Ganho na Recepção
                datum = [numR, numS, numT, numL, numV, retE, ampE, retR, ampR]
                data_line = [f"{datum[i]}" + "\t" for i in range(0, len(datum) - 1)]
                data_line.append(f"{datum[-1]}\n")
                file.writelines(data_line)


def read(file_root):
    file_root = file_root
    with open(file_root) as f:
        # Lê o arquivo .law:
        lines_with_header = f.readlines()
        # Extrai os parâmetros do cabeçalho:
        header = __extract_line_info(lines_with_header[2])
        num_data = len(header)
        lines = lines_with_header[3:]
        lines_data = __lines2numpyarray(lines)

        num_elements = int(np.max(lines_data[:, idx_numV]))  # O número de elementos sendo pulsados
        max_numS = int(np.max(lines_data[:, idx_numS]))
        if max_numS != 0:
            num_shots = int(np.max(lines_data[:, idx_numS]) + 1)  # É um FMC
        elif max_numS == 0:
            num_shots = int(np.max(lines_data[:, idx_numT]) + 1)  # O número de disparos/shots
        else:
            raise ValueError("Valor não suportado para o parâmetro numS no arquivo '.law'")

        delay_law = np.zeros(shape=(num_shots, num_elements), dtype='float')
        amplitude_law = np.zeros(shape=(num_shots, num_elements), dtype='float')

        for shot_idx in range(num_shots):
            i_beg = shot_idx * num_elements
            i_end = (shot_idx + 1) * num_elements

            # Lê todos os parâmetros:
            current_shot_numR = lines_data[i_beg:i_end, idx_numR]  #
            current_shot_numS = lines_data[i_beg:i_end, idx_numS]  #
            current_shot_numT = lines_data[i_beg:i_end, idx_numT]  # Shot
            current_shot_numL = lines_data[i_beg:i_end, idx_numL]  #
            current_shot_numV = lines_data[i_beg:i_end, idx_numV]  # Índice do Emissor
            current_shot_retE = lines_data[i_beg:i_end, idx_retE]  # Lei focal na Emissão
            current_shot_ampE = lines_data[i_beg:i_end, idx_ampE]  # Ganho na Emissão
            current_shot_retR = lines_data[i_beg:i_end, idx_retR]  # Lei focal na Recepção
            current_shot_ampR = lines_data[i_beg:i_end, idx_ampR]  # Ganho na Recepção

            # Assumindo que o delay da transmissão é igual ao da recepção:
            delay_law[shot_idx, :] = current_shot_retE  # Assumindo que current_shot_retE == current_shot_retT

            # Assumindo que a amplitude da transmissão é igual ao da recepção:
            amplitude_law[shot_idx, :] = current_shot_ampE  # Assumindo que current_shot_ampE == current_shot_ampT

    return delay_law, amplitude_law


def __extract_line_info(line):
    return line[:-1].split('\t')


def __lines2numpyarray(lines):
    num_lines = len(lines)
    return np.array([__extract_line_info(lines[i]) for i in range(num_lines)], dtype='float')
