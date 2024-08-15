import typing

if typing.TYPE_CHECKING:
    from pydantic_core import ErrorDetails


class ScoreCardValidationError(ValueError):
    def __init__(self, errs: "typing.List[ErrorDetails]", source_data: dict):
        self.variable_errors = []
        self.other_errors = []
        source_data_vp = source_data.get("variable_params", [])
        for err in errs:
            err_loc = err.get("loc", tuple())
            # Ensuring length is 2 for params and index
            if len(err_loc) >= 2 and err_loc[0] == "variable_params":
                variable_name = "unknown"
                if err_loc[1] >= 0 and err_loc[1] < len(source_data_vp):
                    variable_params = source_data_vp[err_loc[1]]
                    if not isinstance(variable_params, dict):
                        self.other_errors.append(err)
                    else:
                        variable_name = source_data_vp[err_loc[1]].get(
                            "variable", variable_name
                        )
                self.variable_errors.append({"variable_name": variable_name, **err})
            else:
                self.other_errors.append(err)
        super().__init__(
            self._get_error_message(self.variable_errors, self.other_errors)
        )

    @staticmethod
    def _get_error_message(variable_errors, other_errors):
        import json

        lines = [f"Found {len(variable_errors)+len(other_errors)} Validation Errors"]
        if len(variable_errors) > 0:
            lines.append("Variable Errors:")
            for err in variable_errors:
                err_loc = err.get("loc", tuple())
                if len(err_loc) < 2:
                    continue  # This shouldn't happen just as an incase
                location_path = [
                    f"  {err.get('variable_name', 'unknown')} [{err_loc[1]}]"
                ]
                location_path.extend((str(er_loc_it) for er_loc_it in err_loc[2:]))
                lines.append(" -> ".join(location_path))
                if "input" in err:
                    try:
                        input_json = json.dumps(err.get("input"))
                    except TypeError:
                        input_json = err.get("input")
                    lines.append("    Value: " + input_json)
                lines.append("    " + err.get("msg"))
        if len(other_errors) > 0:
            lines.append("Additional Errors:")
            for err in other_errors:
                err_loc = err.get("loc", tuple())
                lines.append(
                    "  " + (" -> ".join((str(er_loc_it) for er_loc_it in err_loc)))
                )
                if "input" in err:
                    try:
                        input_json = json.dumps(err.get("input"))
                    except TypeError:
                        input_json = err.get("input")
                    lines.append("    Value: " + input_json)
                lines.append("    " + err.get("msg"))
                lines.append("    " + err.get("msg"))
        return "\n".join(lines)
