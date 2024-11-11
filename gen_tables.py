import abc
import argparse
import dataclasses
import enum
import itertools
import json
import pathlib
import re
import sqlite3
import sys
import unicodedata
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence, Set
from typing import Any, Literal, TextIO, TypedDict, TypeVar, cast


class ApplicationError(Exception):
    pass


class InvalidRecordError(ApplicationError):
    @property
    def message(self) -> str:
        return cast(str, self.args[0])

    @property
    def line(self) -> int:
        return cast(int, self.args[1])

    def __str__(self) -> str:
        return f"{self.message} at line {self.line}"

    def __init__(self, message: str, line: int) -> None:
        super().__init__(message, line)


def render_uni(cp: int) -> str:
    return f"U+{cp:04X}"


def render_uni_glyphwiki(cp: int) -> str:
    return f"u{cp:04x}"


RE_UNI = re.compile(r"U\+([0-9A-Fa-f]+)$")


def parse_uni(v: str) -> int:
    m = RE_UNI.match(v)
    if m is None:
        raise ValueError(f"invalid unicode representation: {v}")
    return int(m[1], 16)


RE_UNI_GLYPHWIKI = re.compile(r"u([0-9A-Fa-f]{4,6})$")


def parse_uni_glyphwiki(v: str) -> int:
    m = RE_UNI_GLYPHWIKI.match(v)
    if m is None:
        raise ValueError(f"invalid unicode representation: {v}")
    return int(m[1], 16)


RE_UNI_TUPLE_ENVELOPE = re.compile(r"<([^>]+)>$")


def parse_uni_tuple(v: str) -> tuple[int, ...]:
    m = RE_UNI_TUPLE_ENVELOPE.match(v)
    if m is None:
        raise ValueError(f"invalid unicode tuple representation: {v}")

    return tuple(parse_uni(c.strip()) for c in m[1].split(","))


def parse_uni_tuple_glyphwiki(v: str) -> tuple[int, ...]:
    return tuple(parse_uni_glyphwiki(cp) for cp in v.split("-"))


def render_uni_tuple(v: Iterable[int]) -> str:
    return ":".join(render_uni(cp) for cp in v)


def render_uni_tuple_glyphwiki(v: Iterable[int]) -> str:
    return "-".join(render_uni_glyphwiki(cp) for cp in v)


def render_expanded_uni_tuple(v: Iterable[int]) -> tuple[str, ...]:
    return tuple(render_uni(cp) for cp in v)


def parse_uni_set(v: str) -> Set[int]:
    return {parse_uni(c.strip()) for c in v.split(":")}


def parse_uni_tuple_set(v: str) -> Set[tuple[int, ...]]:
    return {parse_uni_tuple(c.strip()) for c in v.split(":")}


def parse_jis_men_ku_ten(v: str) -> int:
    c = 0
    for rn in v.split("-"):
        n = int(rn)
        if n < 1 or n > 94:
            raise ValueError(f"invalid value: {v}")
        c = c * 94 + n - 1

    return c


def is_radical(cp: int) -> bool:
    return (cp >= 0x2F00 and cp <= 0x2FD5) or (cp >= 0x2E80 and cp <= 0x2EFF)


def is_svs(cp: int) -> bool:
    return cp >= 0xFE00 and cp <= 0xFE0F


class JSONMarshallable(metaclass=abc.ABCMeta):
    def __json__(self) -> Any: ...  # pragma: nocover


T1 = TypeVar("T1")
T2 = TypeVar("T2")

UniJisTable = Literal["unijis_90", "unijis_2004"]


def to_pair(i: Iterable[T1]) -> tuple[T1, T1]:
    r = tuple(i)
    if len(r) != 2:
        raise ValueError(f"expecting an iterable that yields a couple of items, got {len(r)} items actually.")

    return cast(tuple[T1, T1], r)


@dataclasses.dataclass
class IVSSVS:
    ivs: tuple[int, int] | None
    svs: tuple[int, int] | None


@dataclasses.dataclass(frozen=True)
class AdobeIVDTableRecord(JSONMarshallable):
    cid: int
    jisx0208_90: int | None
    jisx0208_2004: int | None
    jisx0208_78: int | None
    jisx0208_83: int | None
    jisx0213_2000: int | None
    jisx0213_2004: int | None
    unijis_90: int | None
    unijis_2004: int | None
    uni_ivs: Sequence[tuple[int, int]]

    def __json__(self) -> Any:
        return {
            "cid": self.cid,
            "jisx0208_90": self.jisx0208_90,
            "jisx0208_2004": self.jisx0208_2004,
            "jisx0208_78": self.jisx0208_78,
            "jisx0208_83": self.jisx0208_83,
            "jisx0213_2000": self.jisx0213_2000,
            "jisx0213_2004": self.jisx0213_2004,
            "unijis_90": self.unijis_90,
            "unijis_2004": self.unijis_2004,
            "uni_ivs": self.uni_ivs,
        }

    @classmethod
    def from_dict(cls, d: dict[Any, Any]):
        return cls(
            cid=d["cid"],
            jisx0208_90=d["jisx0208_90"],
            jisx0208_2004=d["jisx0208_2004"],
            jisx0208_78=d["jisx0208_78"],
            jisx0208_83=d["jisx0208_83"],
            jisx0213_2000=d["jisx0213_2000"],
            jisx0213_2004=d["jisx0213_2004"],
            unijis_90=d["unijis_90"],
            unijis_2004=d["unijis_2004"],
            uni_ivs=tuple(tuple(ct) for ct in d["uni_ivs"]),
        )

    def get_unijis_value(self, table: UniJisTable) -> int | None:
        return self.unijis_90 if table == "unijis_90" else self.unijis_2004

    @property
    def ivs_svs(self) -> IVSSVS:
        ivs: tuple[int, int] | None = None
        svs: tuple[int, int] | None = None
        for t in self.uni_ivs:
            if is_svs(t[1]):
                svs = t
            else:
                ivs = t

        return IVSSVS(
            ivs=ivs,
            svs=svs,
        )


class MyJSONEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, JSONMarshallable):
            return cast(Callable[[], Any], o.__json__)()
        return super().default(o)


def do_generate_ivd_table_adobe_japan1(output: TextIO, input: TextIO, **kwargs: Any) -> None:
    records: list[AdobeIVDTableRecord] = []

    def or_none(fn: Callable[[T1], T2], v: T1) -> T2 | None:
        if v == "*":
            return None
        else:
            return fn(v)

    def parse_and_pick_non_radical(v: str) -> int | None:
        cps = [cp for cp in parse_uni_set(v) if not is_radical(cp)]
        return cps[0] if cps else None

    for ln, l in enumerate(input, 1):
        try:
            if l.startswith("#"):
                continue
            l = l.rstrip()
            cols = l.split("\t")

            unijis_90 = or_none(parse_and_pick_non_radical, cols[17])
            unijis_2004 = or_none(parse_and_pick_non_radical, cols[18])

            records.append(
                AdobeIVDTableRecord(
                    cid=int(cols[0]),
                    jisx0208_90=or_none(parse_jis_men_ku_ten, cols[1]),
                    jisx0208_2004=or_none(parse_jis_men_ku_ten, cols[2]),
                    jisx0208_78=or_none(parse_jis_men_ku_ten, cols[3]),
                    jisx0208_83=or_none(parse_jis_men_ku_ten, cols[4]),
                    jisx0213_2000=or_none(parse_jis_men_ku_ten, cols[5]),
                    jisx0213_2004=or_none(parse_jis_men_ku_ten, cols[6]),
                    unijis_90=unijis_90,
                    unijis_2004=unijis_2004,
                    uni_ivs=[cast(tuple[int, int], ct) for ct in parse_uni_tuple_set(cols[19])],
                )
            )
        except ValueError as e:
            raise InvalidRecordError(str(e), ln)

    json.dump(records, output, indent=2, ensure_ascii=False, cls=MyJSONEncoder)


def read_adobe_japan1_ivd_mappings(fp: TextIO) -> Iterable[AdobeIVDTableRecord]:
    for d in json.load(fp):
        yield AdobeIVDTableRecord.from_dict(d)


@dataclasses.dataclass(frozen=True)
class _IVSMappings:
    ivs_tuple_to_ivd_record_mappings: Mapping[int, Mapping[int, AdobeIVDTableRecord]]
    ucs_to_ivd_record_mappings: Mapping[int, AdobeIVDTableRecord]
    ucs_to_ivd_record_mappings_2004: Mapping[int, AdobeIVDTableRecord]

    def lookup_by_ivs_svs(self, sc: tuple[int, ...]) -> AdobeIVDTableRecord:
        if len(sc) == 2:
            sm = self.ivs_tuple_to_ivd_record_mappings.get(sc[1])
            if sm is not None:
                r = sm.get(sc[0])
                if r is not None:
                    return r
        elif len(sc) == 1:
            r = self.ucs_to_ivd_record_mappings_2004.get(sc[0])
            if r is not None:
                return r

        raise ValueError(f"no corresponding IVS mapping found for {render_uni_tuple(sc)}")

    def lookup_by_single_cp(self, cp: int, table: UniJisTable) -> AdobeIVDTableRecord:
        ucs_to_ivd_mappings = (
            self.ucs_to_ivd_record_mappings if table == "unijis_90" else self.ucs_to_ivd_record_mappings_2004
        )
        r = ucs_to_ivd_mappings.get(cp)
        if r is None:
            raise ValueError(f"no corresponding IVS mapping found for {render_uni(cp)}")

        return r


def _build_ivs_mappings(i: Iterable[AdobeIVDTableRecord]) -> _IVSMappings:
    ivs_tuple_to_ivd_record_mappings = defaultdict[int, dict[int, AdobeIVDTableRecord]](dict)
    ucs_to_ivd_record_mappings: dict[int, AdobeIVDTableRecord] = {}
    ucs_to_ivd_record_mappings_2004: dict[int, AdobeIVDTableRecord] = {}

    for r in i:
        for it in r.uni_ivs:
            ivs_tuple_to_ivd_record_mappings[it[1]][it[0]] = r
        if r.unijis_90 is not None:
            ucs_to_ivd_record_mappings[r.unijis_90] = r
        if r.unijis_2004 is not None:
            ucs_to_ivd_record_mappings_2004[r.unijis_2004] = r

    return _IVSMappings(
        ivs_tuple_to_ivd_record_mappings=ivs_tuple_to_ivd_record_mappings,
        ucs_to_ivd_record_mappings=ucs_to_ivd_record_mappings,
        ucs_to_ivd_record_mappings_2004=ucs_to_ivd_record_mappings_2004,
    )


def do_generate_old_new_table(output: TextIO, input: TextIO, ivd_table: TextIO) -> None:
    ivs_svs_mappings: list[tuple[dict[str, tuple[str, ...] | None], dict[str, tuple[str, ...] | None]]] = []

    m = _build_ivs_mappings(read_adobe_japan1_ivd_mappings(ivd_table))

    for ln, l in enumerate(input, 1):
        l = l.rstrip()
        l = re.sub(r"#.*$", "", l)
        if not l:
            continue
        cols = l.split("\t")

        try:
            r = m.lookup_by_ivs_svs(tuple(ord(c) for c in cols[0]))
        except ValueError as e:
            raise InvalidRecordError(f"in the first column, {str(e)}", ln)

        variants: set[AdobeIVDTableRecord] = set()
        for i in range(1, len(cols)):
            try:
                if cols[i]:
                    variants.add(m.lookup_by_ivs_svs(tuple(ord(c) for c in cols[i])))
            except ValueError as e:
                raise InvalidRecordError(f"in the column #{i + 1}, {str(e)}", ln)

        for vr in variants:
            ivs_svs_mappings.append(
                (
                    {
                        "ivs": render_expanded_uni_tuple(vr.ivs_svs.ivs)
                        if vr.ivs_svs.ivs is not None
                        else None,
                        "svs": render_expanded_uni_tuple(vr.ivs_svs.svs)
                        if vr.ivs_svs.svs is not None
                        else None,
                    },
                    {
                        "ivs": render_expanded_uni_tuple(r.ivs_svs.ivs) if r.ivs_svs.ivs is not None else None,
                        "svs": render_expanded_uni_tuple(r.ivs_svs.svs) if r.ivs_svs.svs is not None else None,
                    },
                )
            )

    json.dump(ivs_svs_mappings, output, indent=2, ensure_ascii=False)


class _IVSSVSBase(TypedDict):
    ivs: tuple[str, str] | None
    svs: tuple[str, str] | None
    base90: str | None
    base2004: str | None


def do_generate_table_ivs_svs_base(output: TextIO, input: TextIO) -> None:
    result: list[_IVSSVSBase] = []
    for r in read_adobe_japan1_ivd_mappings(input):
        result.append(
            _IVSSVSBase(
                ivs=cast(tuple[str, str], render_expanded_uni_tuple(r.ivs_svs.ivs))
                if r.ivs_svs.ivs is not None
                else None,
                svs=cast(tuple[str, str], render_expanded_uni_tuple(r.ivs_svs.svs))
                if r.ivs_svs.svs is not None
                else None,
                base90=render_uni(r.unijis_90) if r.unijis_90 is not None else None,
                base2004=render_uni(r.unijis_2004) if r.unijis_2004 is not None else None,
            )
        )

    json.dump(result, output, indent=2, ensure_ascii=False)


@dataclasses.dataclass(frozen=True)
class AsciiTableInfo:
    outlined: bool
    columns_offsets: Sequence[tuple[int, int]]


HORIZONTAL_LINE_CHARS = "-─━┄┅┈┉╌╍═╼╾"
VERTICAL_LINE_CHARS = "|│┃┆┇┊┋╎╏║╽╿"
LEFT_TOP_CORNER_CHARS = "+┌┍┎┏╒╓╔╭"
RIGHT_TOP_CORNER_CHARS = "+┐┑┒┓╕╖╗╮"
LEFT_BOTTOM_CORNER_CHARS = "+└┕┖┗╘╙╚╰"
RIGHT_BOTTOM_CORNER_CHARS = "+┘┘┙┚┛╛╜╝╯"
VERTICAL_AND_LEFT_CHARS = "+┤┥┦┧┨┩┪┫╡╢╣"
VERTICAL_AND_RIGHT_CHARS = "+├┝┞┟┠┡┢┣╞╟╠"
HORIZONTAL_AND_TOP_CHARS = "+┴┵┶┷┸┹┺┻╧╨╩"
HORIZONTAL_AND_BOTTOM_CHARS = "+┬┭┮┯┰┱┲┳╤╥╦"
CROSSING_CHARS = "+|┼┽┾┿╀╁╂╃╄╅╆╇╈╉╊╋╪╫╬"


class LineKind(enum.IntEnum):
    VALUES = 0
    TOP_BORDER = 1
    INTERMEDIATE_BORDER = 2
    BOTTOM_BORDER = 3


AsciiTableRow = tuple[LineKind, Sequence[str] | None]


class AsciiTableReader:
    character_width_getter: Callable[[str], int]
    horizontal_line_char_variation: set[str]
    vertical_line_char_variation: set[str]
    horizontal_and_top_char_variation: set[str]
    horizontal_and_bottom_char_variation: set[str]
    vertical_and_left_char_variation: set[str]
    vertical_and_right_char_variation: set[str]
    crossing_char_variation: set[str]

    def process_first_line(self, l: str) -> tuple[AsciiTableInfo, AsciiTableRow]:
        column_offsets = []
        if l[0] in LEFT_TOP_CORNER_CHARS and l[-1] in RIGHT_TOP_CORNER_CHARS:
            o = self.character_width_getter(l[0])
            i = 1
            ll = len(l) - 1
            while i < ll:
                c = l[i]
                if c not in HORIZONTAL_AND_BOTTOM_CHARS:
                    break
                self.horizontal_and_bottom_char_variation.add(c)
                column_offsets.append((o, o + 1))
                o += self.character_width_getter(c)
                i += 1
            else:
                raise ValueError("invalid table: all the columns are zero-width")

            s = o
            while i < ll:
                c = l[i]
                po = o
                o += self.character_width_getter(c)
                i += 1
                if c in HORIZONTAL_AND_BOTTOM_CHARS:
                    self.horizontal_and_bottom_char_variation.add(c)
                    column_offsets.append((s, po))
                    s = o
                elif l[i] in HORIZONTAL_LINE_CHARS:
                    self.horizontal_line_char_variation.add(c)
                else:
                    raise ValueError("invalid table: expected a horizontal line, got {c}")
            column_offsets.append((s, o))

            return (
                AsciiTableInfo(
                    outlined=True,
                    columns_offsets=column_offsets,
                ),
                (LineKind.TOP_BORDER, None),
            )
        elif l[0] in HORIZONTAL_LINE_CHARS:
            s = o = 0
            i = 0
            while i < len(l):
                c = l[i]
                po = o
                o += self.character_width_getter(c)
                i += 1
                if c in HORIZONTAL_AND_BOTTOM_CHARS:
                    self.horizontal_and_bottom_char_variation.add(c)
                    column_offsets.append((s, po))
                    s = o
                elif c in HORIZONTAL_LINE_CHARS:
                    self.horizontal_line_char_variation.add(c)
                else:
                    raise ValueError("invalid table: expected a horizontal line, got {c}")
            column_offsets.append((s, o))

            return (
                AsciiTableInfo(
                    outlined=False,
                    columns_offsets=column_offsets,
                ),
                (LineKind.TOP_BORDER, None),
            )
        else:
            values = []
            s = o = 0
            i = si = 0
            while i < len(l):
                c = l[i]
                po = o
                o += self.character_width_getter(c)
                i += 1
                if c in VERTICAL_LINE_CHARS:
                    self.vertical_line_char_variation.add(c)
                    column_offsets.append((s, po))
                    values.append(l[si : i - 1])
                    s = o
                    si = i
            column_offsets.append((s, o))
            values.append(l[si:i])

            return (
                AsciiTableInfo(
                    outlined=False,
                    columns_offsets=column_offsets,
                ),
                (LineKind.VALUES, values),
            )

    def process_row(self, ai: AsciiTableInfo, l: str) -> AsciiTableRow:
        def validate_offset_pair(
            ci: int,
            actual_offset_pair: tuple[int, int],
        ) -> None:
            expecting_offset_pair = ai.columns_offsets[ci]
            if not ai.outlined and ci + 1 == len(ai.columns_offsets):
                if expecting_offset_pair[0] == actual_offset_pair[0]:
                    return
            else:
                if expecting_offset_pair == actual_offset_pair:
                    return
            raise ValueError(
                f"column width mismatch at column #{ci}; expected {expecting_offset_pair}, got {actual_offset_pair}"
            )

        o = 0
        i = 0
        ll = len(l)
        line_kind: LineKind
        values: Sequence[str] | None = None
        if ai.outlined:
            if l[0] in VERTICAL_AND_RIGHT_CHARS:
                self.vertical_and_right_char_variation.add(l[0])
                if l[-1] not in VERTICAL_AND_LEFT_CHARS:
                    raise ValueError(f"invalid table: expected a vertical left, got {l[-1]}")
                self.vertical_and_left_char_variation.add(l[-1])
                line_kind = LineKind.INTERMEDIATE_BORDER
            elif l[0] in LEFT_BOTTOM_CORNER_CHARS:
                if l[-1] not in RIGHT_BOTTOM_CORNER_CHARS:
                    raise ValueError(f"invalid table: expected a right bottom corner, got {l[-1]}")
                line_kind = LineKind.BOTTOM_BORDER
            elif l[0] in VERTICAL_LINE_CHARS:
                self.vertical_line_char_variation.add(l[0])
                if l[-1] not in VERTICAL_LINE_CHARS:
                    raise ValueError("invalid table: expected a vertical line")
                self.vertical_line_char_variation.add(l[-1])
                line_kind = LineKind.VALUES
            else:
                raise ValueError(
                    f"invalid table: expected a vertical line, a vertical right, or a left bottom corner, got {l[0]}"
                )
            o = self.character_width_getter(l[0])
            i = 1
            ll -= 1
        else:
            horizontal = l[0] in HORIZONTAL_LINE_CHARS
            crossing = l[0] in CROSSING_CHARS
            if horizontal or crossing:
                line_kind = LineKind.INTERMEDIATE_BORDER
                if horizontal:
                    self.horizontal_line_char_variation.add(l[0])
                elif crossing:
                    self.crossing_char_variation.add(l[0])
                horizontal = l[-1] in HORIZONTAL_LINE_CHARS
                crossing = l[-1] in CROSSING_CHARS
                if not horizontal and not crossing:
                    raise ValueError(f"invalid table: expected a horizontal line or a crossing, got {l[-1]}")
                if horizontal:
                    self.horizontal_line_char_variation.add(l[0])
                elif crossing:
                    self.crossing_char_variation.add(l[0])
            else:
                line_kind = LineKind.VALUES

        s = o
        ci = 0
        match line_kind:
            case LineKind.VALUES:
                si = i
                values = []
                while i <= ll:
                    po = o
                    if i < ll:
                        c = l[i]
                        o += self.character_width_getter(c)
                    else:
                        c = None
                    i += 1

                    if c is None or c in VERTICAL_LINE_CHARS:
                        validate_offset_pair(ci, (s, po))
                        values.append(l[si : i - 1])
                        if c is not None:
                            self.vertical_line_char_variation.add(c)
                        s = o
                        si = i
                        ci += 1
            case LineKind.INTERMEDIATE_BORDER:
                while i <= ll:
                    po = o
                    if i < ll:
                        c = l[i]
                        o += self.character_width_getter(c)
                    else:
                        c = None
                    i += 1

                    if c is None or c in CROSSING_CHARS:
                        validate_offset_pair(ci, (s, po))
                        s = o
                        ci += 1
                    elif c in HORIZONTAL_LINE_CHARS:
                        self.horizontal_line_char_variation.add(c)
                    else:
                        raise ValueError(f"invalid table: expected a horizontal line, got {c}")
            case LineKind.BOTTOM_BORDER:
                while i <= ll:
                    po = o
                    if i < ll:
                        c = l[i]
                        o += self.character_width_getter(c)
                    else:
                        c = None
                    i += 1

                    if c is None or c in HORIZONTAL_AND_TOP_CHARS:
                        validate_offset_pair(ci, (s, po))
                        s = o
                        ci += 1
                    elif c in HORIZONTAL_LINE_CHARS:
                        self.horizontal_line_char_variation.add(c)
                    else:
                        raise ValueError(f"invalid table: expected a horizontal line, got {c}")

        return (line_kind, values)

    def __init__(self, character_width_getter: Callable[[str], int]) -> None:
        self.character_width_getter = character_width_getter
        self.horizontal_line_char_variation = set()
        self.vertical_line_char_variation = set()
        self.horizontal_and_top_char_variation = set()
        self.horizontal_and_bottom_char_variation = set()
        self.vertical_and_left_char_variation = set()
        self.vertical_and_right_char_variation = set()
        self.crossing_char_variation = set()


class AsciiArtTableParseError(Exception):
    line: int | None

    @property
    def message(self) -> str:
        return cast(str, self.args[0])

    def __str__(self) -> str:
        return f"{self.message} at line {self.line}" if self.line is not None else self.message

    def __init__(self, message: str, line: int | None = None) -> None:
        super().__init__(message)
        self.line = line


def read_ascii_art_table(li: Iterator[tuple[int, str]]) -> tuple[Sequence[str], Iterator[Sequence[str]]]:
    l: str
    while True:
        _, l = next(li)
        if l:
            break

    reader = AsciiTableReader(lambda c: 2 if unicodedata.east_asian_width(c) == "W" else 1)
    ai, row = reader.process_first_line(l)
    header_values: Sequence[str] | None = None
    if row[0] == LineKind.VALUES:
        assert row[1] is not None
        header_values = [v.strip() for v in row[1]]

    if header_values is None:
        for ln, l in li:
            try:
                row = reader.process_row(ai, l.rstrip())
            except ValueError as e:
                raise AsciiArtTableParseError(e.args[0], line=ln) from None
            if row[0] == LineKind.VALUES:
                assert row[1] is not None
                header_values = [v.strip() for v in row[1]]
                break
        else:
            raise AsciiArtTableParseError("no header found")

    def __():
        for ln, l in li:
            try:
                row = reader.process_row(ai, l.rstrip())
            except ValueError as e:
                raise AsciiArtTableParseError(e.args[0], line=ln) from None
            if row[0] == LineKind.VALUES:
                assert row[1] is not None
                yield row[1]

    return (header_values, __())


def line_iterator(f: TextIO) -> Iterator[tuple[int, str]]:
    return ((ln, l.rstrip()) for ln, l in enumerate(f, 1))


MINIBATCH_SIZE = 1024
RE_GLYPHWIKI_NAME_U3013 = re.compile(r"u3013\b")
RE_GLYPHWIKI_DUMP_LAST_LINE = re.compile(r"\((\d+) 行\)$")


def do_generate_sqlite3_file_from_glyphwiki_dump(input: TextIO, output: pathlib.Path) -> None:
    def _():
        for pair in line_iterator(input):
            if RE_GLYPHWIKI_DUMP_LAST_LINE.match(pair[1]):
                break
            yield pair

    with sqlite3.connect(output, autocommit=True) as conn:
        conn.execute(
            """CREATE TABLE glyphwiki (name TEXT NOT NULL, related TEXT, data TEXT NOT NULL, PRIMARY KEY (name));"""
        )
        conn.execute("""CREATE INDEX ix_glyphwiki_glyph ON glyphwiki (related);""")
        conn.execute(
            """CREATE INDEX ix_glyphwiki_related ON glyphwiki (data) WHERE data LIKE '99:0:0:0:0:200:200:%';"""
        )

        headers, row_iter = read_ascii_art_table(_())

        indices_map = [headers.index(name) for name in ["name", "related", "data"]]

        row_buf: list[tuple[str, str | None, str]] = []
        for orig_row in itertools.chain(row_iter, [None]):
            row: tuple[str, str | None, str] | None = None
            if orig_row is not None:
                row = cast(tuple[str, str | None, str], tuple(orig_row[i].strip() for i in indices_map))
                # if "u3013" occurs in the "related" column, replace it with None, unless the "name" column specifies the character "u3013" itself.
                if RE_GLYPHWIKI_NAME_U3013.match(row[0]):
                    row_buf.append(row)
                else:
                    row_buf.append((row[0], None if row[1] == "u3013" else row[1], row[2]))
            if len(row_buf) >= MINIBATCH_SIZE or row is None and row_buf:
                old_row_buf = row_buf
                row_buf = []
                conn.execute(
                    f"INSERT INTO glyphwiki VALUES {'(?, ?, ?),' * (len(old_row_buf) - 1)}(?, ?, ?);",
                    [v for row in old_row_buf for v in row],
                )


def build_glyphwiki_alias_data(reference: str) -> str:
    return f"99:0:0:0:0:200:200:{reference}"


CJK_RADICALS_INFOS_OUTPUT_FILE_NAME = "radicals.json"

CJK_RADICALS_CODEPOINT_RANGE = (0x2F00, 0x2FD6)
CJK_RADICALS_SUPPLEMENT_CODEPOINT_RANGE = (0x2E80, 0x2EF4)

RE_GLYPHWIKI_U = re.compile(r"(u[0-9a-f]{4,6})\b")
RE_GLYPHWIKI_GLYPH_DATA_VERBATIM = re.compile(build_glyphwiki_alias_data(r"(u[0-9a-f]{4,6}\b[^$:]*)$"))


def fetch_related_kanji_from_glyphwiki_inner(result: set[str], conn: sqlite3.Connection, name: str, alias_only: bool = False) -> None:
    if name in result:
        return

    result.add(name)

    cur = conn.execute("SELECT related, data FROM glyphwiki WHERE name=?", [name])
    try:
        row = cast(tuple[str | None, str] | None, cur.fetchone())
        if row is None:
            return

        m = RE_GLYPHWIKI_GLYPH_DATA_VERBATIM.match(cast(str, row[1]))
        if m is not None:
            fetch_related_kanji_from_glyphwiki_inner(result, conn, m.group(1), alias_only=alias_only)

        if not alias_only:
            if row[0] is not None:
                m = RE_GLYPHWIKI_U.match(cast(str, row[0]))
                if m is not None:
                    fetch_related_kanji_from_glyphwiki_inner(result, conn, m.group(1), alias_only=False)
    finally:
        cur.close()

    cur = conn.execute("SELECT related, name FROM glyphwiki WHERE data=?", [build_glyphwiki_alias_data(name)])
    try:
        while True:
            row = cur.fetchone()
            if row is None:
                break
            fetch_related_kanji_from_glyphwiki_inner(result, conn, row[1], alias_only=alias_only)

            if not alias_only:
                if row[0] is not None:
                    m = RE_GLYPHWIKI_U.match(cast(str, row[0]))
                    if m is not None:
                        fetch_related_kanji_from_glyphwiki_inner(result, conn, m.group(1), alias_only=False)
    finally:
        cur.close()

def fetch_related_cjk_ideographs_from_glyphwiki(conn: sqlite3.Connection, name: str, alias_only: bool = False) -> Set[str]:
    result: set[str] = set()
    fetch_related_kanji_from_glyphwiki_inner(result, conn, name, alias_only=alias_only)
    result.remove(name)
    return result


def fetch_related_kanji_from_glyphwiki(conn: sqlite3.Connection, cp: int) -> int | None:
    candidates = fetch_related_cjk_ideographs_from_glyphwiki(conn, render_uni_glyphwiki(cp))

    for c in candidates:
        m = RE_GLYPHWIKI_U.match(c)
        assert m is not None
        cp = int(m.group(1)[1:], 16)
        if cp < 0x10000:
            return cp

    return None


def fetch_cjk_radical_infos(
    conn: sqlite3.Connection, codepoint_range: tuple[int, int]
) -> Sequence[tuple[int, int]]:
    retval: list[tuple[int, int]] = []
    for codepoint in range(*codepoint_range):
        related_codepoint = fetch_related_kanji_from_glyphwiki(conn, codepoint)
        if related_codepoint is not None:
            retval.append((codepoint, related_codepoint))

    return retval


def do_generate_table_cjk_radicals(output: TextIO, glyphwiki_db: pathlib.Path) -> None:
    with sqlite3.connect(glyphwiki_db, autocommit=True) as conn:
        json.dump(
            {
                f"U+{codepoint:04X}": f"U+{related_codepoint:04X}"
                for codepoint, related_codepoint in (
                    *fetch_cjk_radical_infos(conn, CJK_RADICALS_CODEPOINT_RANGE),
                    *fetch_cjk_radical_infos(conn, CJK_RADICALS_SUPPLEMENT_CODEPOINT_RANGE),
                )
            },
            output,
            ensure_ascii=False,
            indent=2,
        )


def do_generate_ivd_mapping_table_adobe_japan1_and_hanyo_denshi(
    output: TextIO,
    glyphwiki_db: pathlib.Path,
    ivd_table: TextIO,
) -> None:
    result: list[tuple[tuple[int, int], tuple[int, int]]] = []
    with sqlite3.connect(glyphwiki_db, autocommit=True) as conn:
        for r in read_adobe_japan1_ivd_mappings(ivd_table):
            cp_ivs = r.ivs_svs.ivs
            if cp_ivs is None:
                continue

            corresponding_pairs = fetch_related_cjk_ideographs_from_glyphwiki(conn, render_uni_tuple_glyphwiki(cp_ivs), alias_only=True)

            hdcp: tuple[int, ...] | None = None
            for gr in corresponding_pairs:
                try:
                    _hdcp = parse_uni_tuple_glyphwiki(gr)
                    if len(_hdcp) == 2:
                        hdcp = _hdcp
                        break
                except ValueError:
                    pass

            if hdcp is None:
                if r.unijis_2004 is None:
                    raise ValueError(f"no corresponding Hanyo-Denshi character found for Adobe-Japan1 character {render_uni_tuple(cp_ivs)}; candidates={corresponding_pairs}")
                hdcp = (r.unijis_2004,)

            result.append((cp_ivs, hdcp))

    json.dump([
        {
            "adobe-japan1": render_expanded_uni_tuple(r[0]),
            "hanyo-denshi": render_expanded_uni_tuple(r[1]),
        }
        for r in result
    ], output, ensure_ascii=False, indent=2)


def add_gen_table_ivd_adobe_japan1_subcommand(
    subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]",
) -> argparse.ArgumentParser:
    subparser = subparsers.add_parser("gen-table-ivd-adobe-japan1")
    subparser.add_argument(
        "-i", "--input", type=argparse.FileType("r"), required=True, help="input file (Adobe-Japan1 format)"
    )
    subparser.add_argument("-o", "--output", type=argparse.FileType("w"), required=True, help="output file")
    subparser.set_defaults(
        fn=lambda ns: do_generate_ivd_table_adobe_japan1(
            output=cast(TextIO, ns.output),
            input=cast(TextIO, ns.input),
        ),
    )
    return subparser


def add_gen_table_old_new_subcommand(
    subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]",
) -> argparse.ArgumentParser:
    subparser = subparsers.add_parser("gen-table-old-new")
    subparser.add_argument("-i", "--input", type=argparse.FileType("r"), required=True, help="input file")
    subparser.add_argument("--ivd-table", type=argparse.FileType("r"), required=True, help="IVD table")
    subparser.add_argument("-o", "--output", type=argparse.FileType("w"), required=True, help="output file")
    subparser.set_defaults(
        fn=lambda ns: do_generate_old_new_table(
            output=cast(TextIO, ns.output),
            input=cast(TextIO, ns.input),
            ivd_table=cast(TextIO, ns.ivd_table),
        )
    )
    return subparser


def add_gen_table_ivs_svs_base_subcommand(
    subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]",
) -> argparse.ArgumentParser:
    subparser = subparsers.add_parser("gen-table-ivs-svs-basic")
    subparser.add_argument("-i", "--input", type=argparse.FileType("r"), required=True, help="IVD table")
    subparser.add_argument("-o", "--output", type=argparse.FileType("w"), required=True, help="output file")
    subparser.set_defaults(
        fn=lambda ns: do_generate_table_ivs_svs_base(
            output=cast(TextIO, ns.output),
            input=cast(TextIO, ns.input),
        )
    )
    return subparser


class PathType(argparse.FileType):
    exists: bool

    def __call__(self, value: str):
        p = pathlib.Path(value)
        if self.exists and not p.exists():
            raise FileNotFoundError(p)
        return p

    def __repr__(self):
        return f"{type(self).__name__}(exists={self.exists})"

    def __init__(self, exists: bool = False):
        self.exists = exists


def add_gen_glyphwiki_sqlite3_subcommand(
    subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]",
) -> argparse.ArgumentParser:
    subparser = subparsers.add_parser("gen-glyphwiki-sqlite3")
    subparser.add_argument(
        "-i", "--input", type=argparse.FileType("r"), required=True, help="GlyphWiki dump file"
    )
    subparser.add_argument("-o", "--output", type=PathType(exists=False), required=True, help="output file")
    subparser.set_defaults(
        fn=lambda ns: do_generate_sqlite3_file_from_glyphwiki_dump(
            output=cast(pathlib.Path, ns.output),
            input=cast(TextIO, ns.input),
        )
    )
    return subparser


def add_gen_table_cjk_radicals_subcommand(
    subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]",
) -> argparse.ArgumentParser:
    subparser = subparsers.add_parser("gen-table-cjk-radicals")
    subparser.add_argument(
        "--glyphwiki-db", type=PathType(exists=True), required=True, help="SQLite3 GlyphWiki dump file"
    )
    subparser.add_argument("-o", "--output", type=argparse.FileType("w"), required=True, help="output file")
    subparser.set_defaults(
        fn=lambda ns: do_generate_table_cjk_radicals(
            output=cast(TextIO, ns.output),
            glyphwiki_db=cast(pathlib.Path, ns.glyphwiki_db),
        )
    )
    return subparser


def add_gen_table_ivd_mapping_table_adobe_japan1_and_hanyo_denshi_subcommand(
    subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]",
) -> argparse.ArgumentParser:
    subparser = subparsers.add_parser("gen-table-ivd-mapping-adobe-japan1-and-hanyo-denshi")
    subparser.add_argument("--glyphwiki-db", type=PathType(exists=True), required=True, help="GlyphWiki DB file")
    subparser.add_argument("--ivd-table", type=argparse.FileType("r"), required=True, help="IVD table")
    subparser.add_argument("-o", "--output", type=argparse.FileType("w"), required=True, help="output file")
    subparser.set_defaults(
        fn=lambda ns: do_generate_ivd_mapping_table_adobe_japan1_and_hanyo_denshi(
            output=cast(TextIO, ns.output),
            glyphwiki_db=cast(pathlib.Path, ns.glyphwiki_db),
            ivd_table=cast(TextIO, ns.ivd_table),
        )
    )
    return subparser


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    add_gen_table_ivd_adobe_japan1_subcommand(subparsers)
    add_gen_table_old_new_subcommand(subparsers)
    add_gen_table_ivs_svs_base_subcommand(subparsers)
    add_gen_glyphwiki_sqlite3_subcommand(subparsers)
    add_gen_table_cjk_radicals_subcommand(subparsers)
    add_gen_table_ivd_mapping_table_adobe_japan1_and_hanyo_denshi_subcommand(subparsers)
    return parser


def main(args: Sequence[str]) -> int:
    parser = build_argument_parser()
    ns = parser.parse_args(args[1:])

    if not hasattr(ns, "fn"):
        parser.print_help()
        return 255

    cast(Callable[[argparse.Namespace], None], ns.fn)(ns)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
