"""Minimal GenICam node map: enough of the GenApi XML schema to drive a USB3
Vision camera (the GP3 HD's FLIR Chameleon3) by feature name.

A GenICam camera describes its registers in an XML file stored on the device.
Features (``Width``, ``ExposureTime``, ``AcquisitionStart`` ...) are nodes that
resolve, possibly through other nodes and formulas, to register reads and
writes. Supported nodes: Integer, IntReg, MaskedIntReg, StructReg/StructEntry,
Float, FloatReg, Enumeration, Boolean, Command, String, StringReg,
SwissKnife, IntSwissKnife, Converter and IntConverter. There is no caching:
every read goes to the device. Access modes, invalidators and selectors'
side effects are not modelled (the camera enforces them).

``port`` is any object with ``read(address, length) -> bytes`` and
``write(address, data)``.
"""
from __future__ import annotations

import math
import re
import struct
import xml.etree.ElementTree as ET


class GenICamError(RuntimeError):
    pass


def _tag(el) -> str:
    return el.tag.rsplit("}", 1)[-1]


def _num(text: str):
    t = text.strip()
    try:
        return int(t, 0)
    except ValueError:
        pass
    try:
        return int(t, 10)  # leading zeros ("010")
    except ValueError:
        return float(t)


# ----------------------------------------------------------------- formulas
_TOKEN = re.compile(r"\s*(?:(0x[0-9a-fA-F]+|\d+\.\d*(?:[eE][-+]?\d+)?|\.\d+(?:[eE][-+]?\d+)?"
                    r"|\d+(?:[eE][-+]?\d+)?)|([A-Za-z_][A-Za-z0-9_.]*)"
                    r"|(\*\*|<<|>>|<=|>=|<>|&&|\|\||[-+*/%&|^~!=<>?:(),]))")

_BINARY = {  # operator: (precedence, function)
    "||": (1, lambda a, b: int(bool(a) or bool(b))),
    "&&": (2, lambda a, b: int(bool(a) and bool(b))),
    "|": (3, lambda a, b: int(a) | int(b)),
    "^": (4, lambda a, b: int(a) ^ int(b)),
    "&": (5, lambda a, b: int(a) & int(b)),
    "=": (6, lambda a, b: int(a == b)),
    "<>": (6, lambda a, b: int(a != b)),
    "<": (7, lambda a, b: int(a < b)),
    ">": (7, lambda a, b: int(a > b)),
    "<=": (7, lambda a, b: int(a <= b)),
    ">=": (7, lambda a, b: int(a >= b)),
    "<<": (8, lambda a, b: int(a) << int(b)),
    ">>": (8, lambda a, b: int(a) >> int(b)),
    "+": (9, lambda a, b: a + b),
    "-": (9, lambda a, b: a - b),
    "*": (10, lambda a, b: a * b),
    "/": (10, None),  # integer or float division, see Formula
    "%": (10, lambda a, b: math.fmod(a, b) if isinstance(a, float) or isinstance(b, float)
          else int(math.fmod(a, b))),
    "**": (11, lambda a, b: a ** b),
}

_FUNCS = {
    "SGN": lambda x: (x > 0) - (x < 0), "NEG": lambda x: -x, "ABS": abs,
    "SQRT": math.sqrt, "EXP": math.exp, "LN": math.log, "LG": math.log10,
    "SIN": math.sin, "COS": math.cos, "TAN": math.tan, "ASIN": math.asin,
    "ACOS": math.acos, "ATAN": math.atan, "TRUNC": lambda x: float(math.trunc(x)),
    "FLOOR": lambda x: float(math.floor(x)), "CEIL": lambda x: float(math.ceil(x)),
    "ROUND": lambda x, n=0: round(x, int(n)),
}
_CONSTS = {"PI": math.pi, "E": math.e}


class Formula:
    """A GenApi SwissKnife/Converter formula. ``integer`` selects
    IntSwissKnife semantics (all arithmetic on integers, ``/`` truncates)."""

    def __init__(self, text: str, integer: bool = False):
        self.text, self.integer = text, integer
        self._toks = self._tokenize(text)
        self._i = 0
        self._ast = self._expr(0)
        if self._i != len(self._toks):
            raise GenICamError(f"unexpected {self._toks[self._i][1]!r} in formula {text!r}")

    @staticmethod
    def _tokenize(text):
        out, pos = [], 0
        text = text.strip()
        while pos < len(text):
            m = _TOKEN.match(text, pos)
            if not m or m.end() == pos:
                raise GenICamError(f"cannot parse formula {text!r} at {text[pos:]!r}")
            num, name, op = m.groups()
            out.append(("num", _num(num)) if num else ("name", name) if name else ("op", op))
            pos = m.end()
        return out

    def _peek(self):
        return self._toks[self._i] if self._i < len(self._toks) else (None, None)

    def _take(self, op=None):
        tok = self._peek()
        if op is not None and tok != ("op", op):
            raise GenICamError(f"expected {op!r} in formula {self.text!r}")
        self._i += 1
        return tok

    def _expr(self, min_prec):
        left = self._unary()
        while True:
            kind, op = self._peek()
            if kind != "op":
                break
            if op == "?" and min_prec == 0:
                self._take()
                a = self._expr(0)
                self._take(":")
                b = self._expr(0)
                left = ("?", left, a, b)
                continue
            if op not in _BINARY or _BINARY[op][0] < min_prec or (op == "?"):
                break
            prec = _BINARY[op][0]
            self._take()
            right = self._expr(prec + (0 if op == "**" else 1))  # ** is right-associative
            left = ("bin", op, left, right)
        return left

    def _unary(self):
        kind, val = self._peek()
        if kind == "op" and val in ("-", "+", "~", "!"):
            self._take()
            return ("un", val, self._unary())
        if kind == "op" and val == "(":
            self._take()
            e = self._expr(0)
            self._take(")")
            return e
        if kind == "num":
            self._take()
            return ("num", val)
        if kind == "name":
            self._take()
            if self._peek() == ("op", "(") and val.upper() in _FUNCS:
                self._take()
                args = [self._expr(0)]
                while self._peek() == ("op", ","):
                    self._take()
                    args.append(self._expr(0))
                self._take(")")
                return ("call", val.upper(), args)
            return ("var", val)
        raise GenICamError(f"unexpected end of formula {self.text!r}")

    def __call__(self, variables: dict):
        return self._eval(self._ast, variables)

    def _eval(self, n, v):
        kind = n[0]
        if kind == "num":
            return int(n[1]) if self.integer else n[1]
        if kind == "var":
            if n[1] in v:
                x = v[n[1]]
            elif n[1] in _CONSTS:
                x = _CONSTS[n[1]]
            else:
                raise GenICamError(f"unknown variable {n[1]!r} in formula {self.text!r}")
            return int(x) if self.integer else x
        if kind == "un":
            x = self._eval(n[2], v)
            return {"-": lambda: -x, "+": lambda: x, "~": lambda: ~int(x),
                    "!": lambda: int(not x)}[n[1]]()
        if kind == "?":
            return self._eval(n[2], v) if self._eval(n[1], v) else self._eval(n[3], v)
        if kind == "call":
            r = _FUNCS[n[1]](*(self._eval(a, v) for a in n[2]))
            return int(r) if self.integer else r
        op, a, b = n[1], self._eval(n[2], v), self._eval(n[3], v)
        if op == "/":
            if self.integer:
                return int(a / b) if b else 0
            return a / b
        r = _BINARY[op][1](a, b)
        return int(r) if self.integer else r


# ----------------------------------------------------------------- node map
_REGISTERS = {"IntReg", "MaskedIntReg", "StructEntry", "FloatReg", "StringReg", "Register"}


class NodeMap:
    """Feature access by name: ``get``, ``set``, ``execute``, ``range``,
    ``entries`` (enumeration names) and ``has``."""

    def __init__(self, xml: str | bytes, port):
        self.port = port
        root = ET.fromstring(xml)
        self.nodes: dict[str, ET.Element] = {}
        self._parent: dict[ET.Element, ET.Element] = {}
        self._formulas: dict[tuple, Formula] = {}
        for el in root.iter():
            if _tag(el) == "StructReg":
                for e in el:
                    if _tag(e) == "StructEntry":
                        self._parent[e] = el
            if "Name" in el.attrib and _tag(el) not in ("EnumEntry", "pVariable", "Constant",
                                                       "Expression"):
                self.nodes.setdefault(el.attrib["Name"], el)
        self.model = root.attrib.get("ModelName", "")
        self.vendor = root.attrib.get("VendorName", "")

    # ------------------------------------------------------------ helpers
    def has(self, name: str) -> bool:
        return name in self.nodes

    def _node(self, name: str) -> ET.Element:
        try:
            return self.nodes[name.strip()]
        except KeyError:
            raise GenICamError(f"camera has no feature {name!r}") from None

    def _child(self, el, tag, inherit=True):
        for c in el:
            if _tag(c) == tag:
                return c
        if inherit and el in self._parent:
            return self._child(self._parent[el], tag)
        return None

    def _children(self, el, tag):
        return [c for c in el if _tag(c) == tag]

    def _text(self, el, tag, default=None):
        c = self._child(el, tag)
        return c.text.strip() if c is not None and c.text is not None else default

    def _value_of(self, el, tag, ptag, default=None):
        """Literal <tag> or the value of the node named by <ptag>."""
        p = self._text(el, ptag)
        if p is not None:
            return self.get(p, raw=True)
        t = self._text(el, tag)
        return _num(t) if t is not None else default

    # --------------------------------------------------------- registers
    def _address(self, el) -> int:
        addr = 0
        scopes = [el] + ([self._parent[el]] if el in self._parent else [])
        for scope in scopes:
            for c in scope:
                t = _tag(c)
                if t == "Address":
                    addr += _num(c.text)
                elif t == "pAddress":
                    addr += int(self.get(c.text, raw=True))
                elif t == "IntSwissKnife":
                    addr += int(self._swissknife(c, integer=True))
                elif t == "pIndex":
                    off = c.attrib.get("Offset")
                    if off is None and c.attrib.get("pOffset"):
                        off = self.get(c.attrib["pOffset"], raw=True)
                    addr += int(self.get(c.text, raw=True)) * (_num(off) if isinstance(off, str)
                                                               else int(off or 1))
        return addr

    def _reg_spec(self, el):
        length = int(self._value_of(el, "Length", "pLength", 4))
        big = self._text(el, "Endianess", "LittleEndian") == "BigEndian"
        return self._address(el), length, big

    def _read_int(self, el) -> int:
        addr, length, big = self._reg_spec(el)
        raw = int.from_bytes(bytes(self.port.read(addr, length)), "big" if big else "little")
        nbits = 8 * length
        lo, hi = self._bits(el, nbits, big)
        if lo is not None:
            raw = (raw >> lo) & ((1 << (hi - lo + 1)) - 1)
            nbits = hi - lo + 1
        if self._text(el, "Sign", "Unsigned") == "Signed" and raw >> (nbits - 1):
            raw -= 1 << nbits
        return raw

    def _write_int(self, el, value: int):
        addr, length, big = self._reg_spec(el)
        order = "big" if big else "little"
        value = int(value)
        lo, hi = self._bits(el, 8 * length, big)
        if lo is not None:
            cur = int.from_bytes(bytes(self.port.read(addr, length)), order)
            mask = ((1 << (hi - lo + 1)) - 1) << lo
            value = (cur & ~mask) | ((value << lo) & mask)
        value &= (1 << (8 * length)) - 1
        self.port.write(addr, value.to_bytes(length, order))

    def _bits(self, el, nbits, big):
        """(lo, hi) bit shifts of a masked field, or (None, None)."""
        bit = self._text(el, "Bit")
        if bit is not None:
            lsb = msb = int(bit)
        else:
            lsb, msb = self._text(el, "LSB"), self._text(el, "MSB")
            if lsb is None and msb is None:
                return None, None
            lsb = int(lsb) if lsb is not None else (nbits - 1 if big else 0)
            msb = int(msb) if msb is not None else (0 if big else nbits - 1)
        if big:  # GenICam numbers bits of big-endian registers from the MSB
            lsb, msb = nbits - 1 - lsb, nbits - 1 - msb
        return min(lsb, msb), max(lsb, msb)

    # ---------------------------------------------------------- formulas
    def _formula(self, el, tag, integer):
        key = (id(el), tag)
        if key not in self._formulas:
            self._formulas[key] = Formula(self._text(el, tag, "0"), integer)
        return self._formulas[key]

    def _variables(self, el) -> dict:
        v = {}
        for c in el:
            t = _tag(c)
            if t == "pVariable":
                v[c.attrib["Name"]] = self.get(c.text, raw=True)
            elif t == "Constant":
                v[c.attrib["Name"]] = _num(c.text)
        for c in el:  # expressions may use the variables and constants above
            if _tag(c) == "Expression":
                v[c.attrib["Name"]] = Formula(c.text)(v)
        return v

    def _swissknife(self, el, integer):
        return self._formula(el, "Formula", integer)(self._variables(el))

    # ------------------------------------------------------------- public
    def get(self, name: str, raw: bool = False):
        """Feature value. Enumerations return the entry name (its integer with
        raw=True), Booleans a bool, Commands whether they are still running."""
        el = self._node(name)
        t = _tag(el)
        if t in ("Integer", "Float"):
            v = self._value_of(el, "Value", "pValue", 0)
            return int(v) if t == "Integer" else float(v)
        if t in ("IntReg", "MaskedIntReg", "StructEntry"):
            return self._read_int(el)
        if t == "FloatReg":
            addr, length, big = self._reg_spec(el)
            fmt = ("<>"[big]) + ("f" if length == 4 else "d")
            return struct.unpack(fmt, bytes(self.port.read(addr, length)))[0]
        if t == "Enumeration":
            v = int(self._value_of(el, "Value", "pValue", 0))
            if raw:
                return v
            for e in self._children(el, "EnumEntry"):
                if int(self._value_of(e, "Value", "pValue", -1)) == v:
                    return e.attrib["Name"]
            return v
        if t == "Boolean":
            v = self._value_of(el, "Value", "pValue", 0)
            on = _num(self._text(el, "OnValue", "1"))
            return int(v) if raw else v == on
        if t == "Command":
            v = self._value_of(el, "Value", "pValue", 0)
            cv = self._value_of(el, "CommandValue", "pCommandValue", 1)
            return v == cv
        if t in ("SwissKnife", "IntSwissKnife"):
            return self._swissknife(el, integer=t == "IntSwissKnife")
        if t in ("Converter", "IntConverter"):
            v = self._variables(el)
            v["TO"] = self.get(self._text(el, "pValue"), raw=True)
            r = self._formula(el, "FormulaFrom", t == "IntConverter")(v)
            return int(r) if t == "IntConverter" else float(r)
        if t == "StringReg":
            addr, length, _ = self._reg_spec(el)
            return bytes(self.port.read(addr, length)).split(b"\0", 1)[0].decode("ascii", "replace")
        if t == "String":
            p = self._text(el, "pValue")
            return self.get(p) if p else self._text(el, "Value", "")
        raise GenICamError(f"cannot read {t} node {name!r}")

    def set(self, name: str, value):
        el = self._node(name)
        t = _tag(el)
        if t in ("Integer", "Float"):
            p = self._text(el, "pValue")
            if p is None:
                v = self._child(el, "Value")
                if v is None:
                    raise GenICamError(f"{name} is read-only")
                v.text = str(value)
                return
            return self.set(p, int(value) if t == "Integer" else value)
        if t in ("IntReg", "MaskedIntReg", "StructEntry"):
            return self._write_int(el, value)
        if t == "FloatReg":
            addr, length, big = self._reg_spec(el)
            fmt = ("<>"[big]) + ("f" if length == 4 else "d")
            return self.port.write(addr, struct.pack(fmt, float(value)))
        if t == "Enumeration":
            if isinstance(value, str):
                for e in self._children(el, "EnumEntry"):
                    if e.attrib["Name"] == value:
                        value = self._value_of(e, "Value", "pValue")
                        break
                else:
                    raise GenICamError(f"{name} has no entry {value!r} "
                                       f"(entries: {', '.join(self.entries(name))})")
            return self.set(self._text(el, "pValue"), int(value))
        if t == "Boolean":
            v = _num(self._text(el, "OnValue" if value else "OffValue", "1" if value else "0"))
            return self.set(self._text(el, "pValue"), v)
        if t in ("Converter", "IntConverter"):
            v = self._variables(el)
            v["FROM"] = value
            r = self._formula(el, "FormulaTo", t == "IntConverter")(v)
            return self.set(self._text(el, "pValue"), r)
        if t == "Command":
            return self.execute(name)
        raise GenICamError(f"cannot write {t} node {name!r}")

    def execute(self, name: str):
        el = self._node(name)
        if _tag(el) != "Command":
            raise GenICamError(f"{name} is not a command")
        cv = self._value_of(el, "CommandValue", "pCommandValue", 1)
        self.set(self._text(el, "pValue"), cv)

    def range(self, name: str) -> tuple:
        """(min, max, increment) of an Integer/Float feature (None if unknown)."""
        el = self._node(name)
        t = _tag(el)
        lo = self._value_of(el, "Min", "pMin")
        hi = self._value_of(el, "Max", "pMax")
        inc = self._value_of(el, "Inc", "pInc")
        p = self._text(el, "pValue")
        if (lo is None or hi is None) and p and _tag(self._node(p)) in ("Integer", "Float"):
            plo, phi, pinc = self.range(p)
            lo = plo if lo is None else lo
            hi = phi if hi is None else hi
            inc = pinc if inc is None else inc
        return lo, hi, inc

    def entries(self, name: str) -> list[str]:
        """Names of an Enumeration's entries."""
        return [e.attrib["Name"] for e in self._children(self._node(name), "EnumEntry")]

    def try_set(self, name: str, value) -> bool:
        """set() that returns False instead of raising (optional features)."""
        try:
            if not self.has(name):
                return False
            self.set(name, value)
            return True
        except Exception:
            return False
