"""
Authors:    Shichao Wang
Created at: 2021-04-11 15:57:20
"""
from argparse import ArgumentParser
from dataclasses import MISSING, Field, fields, is_dataclass
from typing import Dict, Mapping, Type, TypeVar, Union

DataClass = TypeVar("DataClass")
DataClassType = Type[DataClass]


def parse_into_dataclass(
    dataclass_type: Union[DataClassType, Mapping[str, DataClassType]]
) -> DataClass:
    if isinstance(dataclass_type, Mapping):
        parser = ArgumentParser()
        subparsers = parser.add_subparsers(dest="cmd")
        for cmd, dataclass_type in dataclass_type.items():
            sub_parser = subparsers.add_parser(cmd)
            make_argument_parser(dataclass_type, sub_parser)

        args, _ = parser.parse_known_args()
        dataclass_type = dataclass_type[args.cmd]
        delattr(args, "cmd")
        return dataclass_type(**vars(args))
    else:
        return parse_into_a_dataclass(dataclass_type)


def parse_into_a_dataclass(dataclass_type: Type[DataClass]) -> DataClass:
    if not is_dataclass(dataclass_type):
        raise ValueError()
    parser = make_argument_parser(dataclass_type)
    args, _ = parser.parse_known_args()
    return dataclass_type(**vars(args))


def complete_default_and_required(kwargs: Dict, dataclass_field: Field):
    new_kwargs = kwargs.copy()
    # del kwargs

    if new_kwargs.get("required", False):  # default is not required
        del new_kwargs["default"]
    elif (
        dataclass_field.default is not MISSING
        or dataclass_field.default_factory is not MISSING
    ):
        new_kwargs["default"] = (
            dataclass_field.default
            if dataclass_field.default_factory is MISSING
            else dataclass_field.default_factory()
        )
        new_kwargs.setdefault("help", "")
        new_kwargs["help"] += "DEFAULT is %s" % str(new_kwargs["default"])
    else:
        new_kwargs["required"] = True
    return new_kwargs


def make_argument_parser(
    dataclass_type: Type[DataClass], parser: ArgumentParser = None
) -> ArgumentParser:
    parser = parser or ArgumentParser(
        getattr(dataclass_type, "__program__", dataclass_type.__name__)
    )
    field: Field
    for field in fields(dataclass_type):
        if not field.init:
            continue

        kwargs = field.metadata.copy()
        kwargs["type"] = field.type
        kwargs = complete_default_and_required(kwargs, field)
        parser.add_argument(f"--{field.name}", **kwargs)

    return parser
