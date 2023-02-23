import torch

symbol_placeholder_dict = {
    'u': -1,
    'ap': -3,
    '=': -4,
    '<': -5,
    '>': -6,
}
placeholder_symbol_dict = {value: key for key, value in symbol_placeholder_dict.items()}


def flatten_structure(query_structure):
    if type(query_structure) == str:
        return [query_structure]

    flat_structure = []
    for element in query_structure:
        flat_structure.extend(flatten_structure(element))

    return flat_structure


def query_to_atoms(query_structure, flat_ids):
    flat_structure = flatten_structure(query_structure)
    batch_size, query_length = flat_ids.shape
    assert len(flat_structure) == query_length

    query_triples = []
    variable = 0
    previous = flat_ids[:, 0]
    conjunction_mask = []
    attr_mask = []
    filters = []
    no_head_entity = False

    start_index = 1

    for i in range(start_index, query_length):
        if flat_structure[i] == 'r':
            variable -= 1
            triples = torch.empty(batch_size, 3, device=flat_ids.device, dtype=torch.long)
            triples[:, 0] = previous
            triples[:, 1] = flat_ids[:, i]
            triples[:, 2] = variable

            query_triples.append(triples)
            previous = variable
            conjunction_mask.append(True)
            attr_mask.append(False)
            filters.append(None)
        elif flat_structure[i] == 'a':
            if i + 2 <= query_length and flat_structure[i+1] == 'v':
                if i == 1:
                    no_head_entity = True
                    previous = variable
                    variable -= 1
                previous = variable  # only need the tail variable
                triples = torch.empty(batch_size, 3, device=flat_ids.device, dtype=torch.long)
                triples[:, 0] = previous
                triples[:, 1] = flat_ids[:, i]
                triples[:, 2] = variable

                query_triples.append(triples)
                attr_mask.append(True)
                conjunction_mask.append(True)

                i += 1
                filter = torch.empty(batch_size, 2, device=flat_ids.device, dtype=torch.float)
                filter[:, 0] = flat_ids[:, i]  # value
                filter[:, 1] = flat_ids[:, i+1]  # filter expression
                filters.append(filter)
                i += 1
            else:
                # attribute value prediction
                variable -= 1
                triples = torch.empty(batch_size, 3, device=flat_ids.device, dtype=torch.long)
                triples[:, 0] = previous
                triples[:, 1] = flat_ids[:, i]
                triples[:, 2] = variable

                previous = variable
                query_triples.append(triples)
                attr_mask.append(True)
                conjunction_mask.append(True)
                filters.append(None)
        elif flat_structure[i] == 'e':
            previous = flat_ids[:, i]
            variable += 1
        elif flat_structure[i] == 'u':
            conjunction_mask = [False] * len(conjunction_mask)

    atoms = torch.stack(query_triples, dim=1)
    num_variables = variable * -1
    conjunction_mask = torch.tensor(conjunction_mask).unsqueeze(0).expand(batch_size, -1)
    attr_mask = torch.tensor(attr_mask).unsqueeze(0).expand(batch_size, -1)

    return atoms, num_variables, conjunction_mask, attr_mask, filters, no_head_entity


query_name_dict = {('e', ('r',)): '1p',
                   ('e', ('ap', 'a')): '1ap',
                   ('e', ('dp',)): '1dp',
                   ('e', ('r', 'r')): '2p',
                   (('e', ('r',)), ('ap', 'a')): '2ap',
                   ('e', ('r', 'r', 'r',)): '3p',
                   (('e', ('r', 'r',)), ('ap', 'a')): '3ap',
                   (('e', ('r',)), ('e', ('r',))): '2i',
                   (('e', ('r',)), ('e', ('r',)), ('e', ('r',))): '3i',
                   ((('e', ('r',)), ('e', ('r',))), ('r',)): 'ip',
                   (('e', ('r', 'r')), ('e', ('r',))): 'pi',
                   (('dp',), ('dv', '=')): 'di',
                   (('ap', 'a'), ('v', 'f')): 'ai',
                   (('ap', 'a'), ('v', '=')): 'ai-eq',
                   (('ap', 'a'), ('v', '<')): 'ai-lt',
                   (('ap', 'a'), ('v', '>')): 'ai-gt',
                   ((('ap', 'a'), ('v', 'f')), (('ap', 'a'), ('v', 'f'))): '2ai',
                   (('e', ('r',)), (('ap', 'a'), ('v', 'f'))): 'pai',
                   ((('ap', 'a'), ('v', 'f')), ('r')): 'aip',
                   (('e', ('r',)), ('e', ('r',)), ('u',)): '2u',
                   ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)): 'up',
                   ((('ap', 'a'), ('v', 'f')), (('ap', 'a'), ('v', 'f')), ('u',)): 'au',
                   }
name_query_dict = {value: key for key, value in query_name_dict.items()}
