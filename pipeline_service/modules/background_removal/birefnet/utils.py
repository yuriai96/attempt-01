def check_state_dict(state_dict, unwanted_prefixes=("module.", "_orig_mod.")):
    for key in list(state_dict.keys()):
        new_key = key
        for prefix in unwanted_prefixes:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        if new_key != key:
            state_dict[new_key] = state_dict.pop(key)
    return state_dict
