from domyn_eval.drivers.stat.base import Sample


def normal_sample(**fields: object) -> Sample:
    """A sample that is fine unless ``fields`` say otherwise."""
    return Sample.model_validate({"truncated": False, "looping": False, "scored": True, **fields})
