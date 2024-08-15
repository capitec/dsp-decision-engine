class SpockFlowException(Exception): ...


class RequiredOptionalDependencyError(SpockFlowException):
    def __init__(self, requires, install_name):
        self.requires = requires
        self.install_name = install_name

    def __repr__(self):
        return (
            f"Missing optional dependency '{self.requires}'."
            f"Use pip or conda to install {self.install_name}."
        )
