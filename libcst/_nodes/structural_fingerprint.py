# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Provides the implementation of `CSTNode.structural_fingerprint`.

This module computes a deterministic, cross-process-stable, structure-based
fingerprint of a CSTNode tree. Unlike Python's builtin `hash()` which is
identity-based for CSTNodes and PYTHONHASHSEED-dependent for primitives,
the fingerprint returned here is stable across:

    * Different Python interpreter invocations (different PYTHONHASHSEED)
    * Different object identities representing equivalent structure
    * Serialization round trips (e.g. pickle)
    * Visitor passes that do not modify structure

The implementation uses FNV-1a (64-bit) combined with incremental mixing so
that at no point do we rely on Python's built-in ``hash()``, nor the implicit
hashing of ``str``, ``bytes``, or ``tuple`` via ``__hash__``.

Fingerprint semantics align exactly with ``deep_equals``:
    * Two nodes ``deep_equals(a, b)`` == True must produce the same fingerprint.
    * The reverse (same fingerprint implies deep_equals) is not guaranteed
      (collisions are allowed, though the 64-bit output space makes them rare).
"""

from dataclasses import fields
from typing import Sequence

from libcst._maybe_sentinel import MaybeSentinel
from libcst._nodes.base import CSTNode

# ---------------------------------------------------------------------------
# Deterministic integer mixing / hashing primitives
#
# We use FNV-1a 64-bit as the base primitive because it is:
#   * Simple to implement correctly
#   * Stable across platforms / Python versions
#   * Free of any table-lookup / data-dependent branching
#   * Distributes well enough for our use case (structural fingerprinting)
#
# We combine multiple hashes with an additional mixing step to avoid trivial
# collisions (e.g. concatenation of two inputs vs one combined input).
# ---------------------------------------------------------------------------

_FNV1A_64_OFFSET = 14695981039346656037
_FNV1A_64_PRIME = 1099511628211
_MASK64 = (1 << 64) - 1


def _fnv1a_64_start() -> int:
    return _FNV1A_64_OFFSET


def _fnv1a_64_update(state: int, data: bytes) -> int:
    h = state & _MASK64
    for b in data:
        h ^= b
        h = (h * _FNV1A_64_PRIME) & _MASK64
    return h


def _fnv1a_64_update_int(state: int, value: int) -> int:
    """
    Mix a bounded integer into the hash state.

    To avoid platform / Python int-width issues we always encode a signed
    64-bit wide integer in little-endian. For values outside that range we
    also append the big-endian byte length and the raw bytes of the value.
    """
    # Fast path for the common case: small ints fit in 8 bytes
    try:
        data = value.to_bytes(8, byteorder="little", signed=True)
        h = _fnv1a_64_update(state, data)
        # If the round-trip decodes correctly we are done
        roundtrip = int.from_bytes(data, byteorder="little", signed=True)
        if roundtrip == value:
            return h
    except (OverflowError, ValueError):
        pass
    # Slow path: encode length + big-endian bytes
    sign = 1 if value >= 0 else -1
    absval = value * sign
    length = (absval.bit_length() + 7) // 8
    data = b"\x00" if sign > 0 else b"\x01"
    data += length.to_bytes(4, byteorder="little", signed=False)
    data += absval.to_bytes(length, byteorder="big", signed=False)
    return _fnv1a_64_update(state, data)


def _hash_combine(parts: Sequence[int]) -> int:
    """
    Combine several 64-bit hash values into a single 64-bit value using
    incremental FNV-1a mixing. This is used for e.g. combining field hashes
    or sequence element hashes.
    """
    h = _fnv1a_64_start()
    for p in parts:
        h = _fnv1a_64_update_int(h, p)
    return h


# ---------------------------------------------------------------------------
# Type tag encoding
#
# To ensure that structurally identical but differently-typed nodes produce
# different fingerprints, we encode a type identifier as the combination of
# (module fully-qualified-name, qualified class name).
#
# We also tag atomic value types (None, bool, int, float, str, bytes,
# MaybeSentinel) with a unique integer tag so that e.g. the integer ``1`` and
# the string ``"1"`` cannot collide.
# ---------------------------------------------------------------------------

_TAG_NONE = 1
_TAG_BOOL = 2
_TAG_INT = 3
_TAG_FLOAT = 4
_TAG_COMPLEX = 5
_TAG_STR = 6
_TAG_BYTES = 7
_TAG_ELLIPSIS = 8
_TAG_MAYBE_SENTINEL = 9
_TAG_CST_NODE = 10
_TAG_SEQUENCE = 11
_TAG_OPTIONAL_NONE = 12


def _encode_type_tag(type_obj: type) -> int:
    """
    Produce a stable, cross-process fingerprint for a Python type.

    The fingerprint is based on the module's fully-qualified name and the
    class' ``__qualname__`` (so that inner classes are distinguished from
    top-level ones). Types defined in different modules but with identical
    class names will therefore produce different fingerprints.
    """
    module = getattr(type_obj, "__module__", "") or ""
    qualname = getattr(type_obj, "__qualname__", type_obj.__name__) or type_obj.__name__

    h = _fnv1a_64_start()
    h = _fnv1a_64_update(h, module.encode("utf-8"))
    h = _fnv1a_64_update(h, b"\x00")  # separator; avoids ambiguity
    h = _fnv1a_64_update(h, qualname.encode("utf-8"))
    return h


# ---------------------------------------------------------------------------
# Leaf / atomic value fingerprinting
#
# The ordering here is critical!  ``bool`` is a subclass of ``int`` in Python
# so we MUST check for ``bool`` BEFORE checking for ``int``. Similarly we
# must handle ``None``, ``str``, and ``bytes`` explicitly to avoid them
# being mis-dispatchd as Sequence or something else.
# ---------------------------------------------------------------------------


def _fingerprint_none() -> int:
    h = _fnv1a_64_start()
    h = _fnv1a_64_update_int(h, _TAG_NONE)
    return h


def _fingerprint_bool(value: bool) -> int:
    h = _fnv1a_64_start()
    h = _fnv1a_64_update_int(h, _TAG_BOOL)
    h = _fnv1a_64_update_int(h, 1 if value else 0)
    return h


def _fingerprint_int(value: int) -> int:
    h = _fnv1a_64_start()
    h = _fnv1a_64_update_int(h, _TAG_INT)
    h = _fnv1a_64_update_int(h, value)
    return h


def _fingerprint_float(value: float) -> int:
    import struct

    h = _fnv1a_64_start()
    h = _fnv1a_64_update_int(h, _TAG_FLOAT)
    # IEEE-754 double precision. NaN canonicalization: treat all NaNs
    # identically so that float("nan") deep_equals float("nan") semantics
    # hold in fingerprint space.
    if value != value:  # NaN
        data = b"\x7f\xf8\x00\x00\x00\x00\x00\x00"  # quiet NaN
    else:
        data = struct.pack("<d", value)
    h = _fnv1a_64_update(h, data)
    return h


def _fingerprint_complex(value: complex) -> int:
    h = _fnv1a_64_start()
    h = _fnv1a_64_update_int(h, _TAG_COMPLEX)
    h = _hash_combine([h, _fingerprint_float(value.real), _fingerprint_float(value.imag)])
    return h


def _fingerprint_str(value: str) -> int:
    h = _fnv1a_64_start()
    h = _fnv1a_64_update_int(h, _TAG_STR)
    # Encode length as well as bytes so that a prefix-identical pair cannot
    # collide e.g. "a" + "bc" vs a different string.
    encoded = value.encode("utf-8")
    h = _fnv1a_64_update_int(h, len(encoded))
    h = _fnv1a_64_update(h, encoded)
    return h


def _fingerprint_bytes(value: bytes) -> int:
    h = _fnv1a_64_start()
    h = _fnv1a_64_update_int(h, _TAG_BYTES)
    h = _fnv1a_64_update_int(h, len(value))
    h = _fnv1a_64_update(h, value)
    return h


def _fingerprint_ellipsis() -> int:
    h = _fnv1a_64_start()
    h = _fnv1a_64_update_int(h, _TAG_ELLIPSIS)
    return h


def _fingerprint_maybe_sentinel(value: MaybeSentinel) -> int:
    h = _fnv1a_64_start()
    h = _fnv1a_64_update_int(h, _TAG_MAYBE_SENTINEL)
    # Use the enum's name (e.g. "DEFAULT") as stable discriminator.
    name = value.name
    encoded = name.encode("utf-8")
    h = _fnv1a_64_update_int(h, len(encoded))
    h = _fnv1a_64_update(h, encoded)
    return h


# ---------------------------------------------------------------------------
# Recursive fingerprinting dispatcher
# ---------------------------------------------------------------------------


def structural_fingerprint(value: object) -> int:
    """
    Compute a deterministic, cross-process-stable structural fingerprint of
    a CST value.

    The returned integer is always in the range of an unsigned 64-bit value
    (``0 .. 2**64 - 1``) and never depends on Python's built-in ``hash()``,
    object identity, or the current ``PYTHONHASHSEED``.

    Semantics follow :func:`deep_equals` exactly:

        * Fields whose dataclass metadata ``compare`` is ``False`` are
          excluded from the fingerprint (these hold internal state /
          metadata / caches).
        * ``Sequence`` fields (including empty sequences) are compared
          element-by-element.
        * ``Optional[...]`` fields that are ``None`` participate as a
          ``None`` tag.
        * ``MaybeSentinel`` values are compared by their enum member name.
        * Nested CST nodes are fingerprinted recursively.
    """
    # Dispatch must be done in a very specific order to avoid aliasing types
    # that stand in a subclass / isinstance relation to each other.

    # 1) Singletons whose identity is their whole meaning
    if value is None:
        return _fingerprint_none()
    if value is ...:
        return _fingerprint_ellipsis()

    # 2) bool BEFORE int, because isinstance(True, int) is True
    if isinstance(value, bool):
        return _fingerprint_bool(value)

    # 3) int
    if isinstance(value, int):
        return _fingerprint_int(value)

    # 4) float
    if isinstance(value, float):
        return _fingerprint_float(value)

    # 5) complex
    if isinstance(value, complex):
        return _fingerprint_complex(value)

    # 6) str and bytes before Sequence check -- they are Sequences but we
    #    want dedicated handling.
    if isinstance(value, str):
        return _fingerprint_str(value)
    if isinstance(value, bytes):
        return _fingerprint_bytes(value)

    # 7) MaybeSentinel -- an Enum, before the generic Sequence path
    if isinstance(value, MaybeSentinel):
        return _fingerprint_maybe_sentinel(value)

    # 8) CSTNode -- the main event
    if isinstance(value, CSTNode):
        return _structural_fingerprint_cst_node(value)

    # 9) Generic Sequence (list/tuple/...), excluding str/bytes which we
    #    already handled above.
    if isinstance(value, Sequence):
        return _structural_fingerprint_sequence(value)

    # 10) Fallback: unknown type. To guarantee that we never silently rely
    #     on the builtin hash, we raise here rather than dropping through
    #     to ``hash()`` or ``id()``. Callers that need support for more
    #     atomic types should extend the dispatcher above.
    raise TypeError(
        f"structural_fingerprint is not defined for values of type "
        f"{type(value).__module__}.{type(value).__qualname__!r}"
    )


def _structural_fingerprint_sequence(seq: Sequence[object]) -> int:
    """
    Fingerprint a generic Sequence (list, tuple, custom sequence, ...)
    element by element. The resulting hash encodes the length and each
    element's fingerprint so that sequences of different length, or whose
    elements differ, produce distinct fingerprints.
    """
    h = _fnv1a_64_start()
    h = _fnv1a_64_update_int(h, _TAG_SEQUENCE)
    h = _fnv1a_64_update_int(h, len(seq))
    element_fps = [structural_fingerprint(el) for el in seq]
    if element_fps:
        h = _hash_combine([h, *element_fps])
    return h


def _structural_fingerprint_cst_node(node: "CSTNode") -> int:
    """
    Fingerprint a single CSTNode by combining:

        * A unique type tag (module + qualname)
        * For each dataclass field with ``compare=True``:
              - the field name (ordered as per dataclasses.fields())
              - the field value's fingerprint

    Fields whose ``compare`` metadata is ``False`` are deliberately
    excluded so that internal state (caches, metadata wrappers, etc.)
    does not influence the fingerprint.
    """
    h = _fnv1a_64_start()
    h = _fnv1a_64_update_int(h, _TAG_CST_NODE)
    type_tag = _encode_type_tag(type(node))
    h = _hash_combine([h, type_tag])

    for f in fields(node):
        if not f.compare:
            continue
        field_name_bytes = f.name.encode("utf-8")
        field_name_fp_start = _fnv1a_64_start()
        field_name_fp = _fnv1a_64_update(field_name_fp_start, field_name_bytes)
        value_fp = structural_fingerprint(getattr(node, f.name))
        combined = _hash_combine([field_name_fp, value_fp])
        h = _hash_combine([h, combined])

    return h
