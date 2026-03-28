from dataclasses import dataclass


class Config:
    def __init__(self, args: dict[str, float | str], defaults: dict[str, float | str]):
        self.args = args
        self.defaults = defaults


    def __try_convert(self, value):
        try:
            value = float(value)
            return int(value) if value.is_integer() else value
        except:
            if value == "True":
                return True
            elif value == "False":
                return False
            return value


    def __get(self, name):
        value = self.args.get(name, self.defaults[name])
        if isinstance(value, list):
            return [self.__try_convert(val) for val in value]
        return self.__try_convert(value)


    def __getitem__(self, name):
        return self.__get(name)


    def __getattr__(self, name):
        return self.__get(name)
