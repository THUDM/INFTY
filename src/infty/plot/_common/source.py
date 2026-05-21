def resolve_source_name(source=None, source_name=None, default="custom"):
    if source_name:
        return str(source_name)
    if source is None:
        return default
    return str(getattr(source, "name", source.__class__.__name__.lower()))


def resolve_conflict_records(source=None, conflict_records=None):
    if conflict_records is not None:
        return conflict_records
    if source is None:
        return None
    return getattr(source, "conflict_records", None)


def resolve_similarity_values(source=None, sim_values=None):
    if sim_values is not None:
        return sim_values
    if source is None:
        return None

    values = getattr(source, "sim_list", None)
    if values is None:
        values = getattr(source, "sim_arr", None)
    return values
