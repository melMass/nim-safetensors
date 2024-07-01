import strutils, tables

type
  DataType* = enum
    BOOL, U8, I8, F8_E5M2, F8_E4M3, I16, U16, F16, BF16,
    I32, U32, F32, F64, I64, U64

  TensorHeader* = object
    dtype*: DataType
    shape*: seq[int]
    data_offsets*: tuple[start: int, `end`: int]
    # data_offsets: seq[int]

  Safetensor* = ref object
    metadata*: Table[string, string]
    tensors*: Table[string, TensorHeader]
    data*: seq[byte]

proc toDataType*(s: string): DataType =
  case s.toUpper()
  of "F16": F16
  of "F32": F32
  of "I8": I8
  of "U8": U8
  of "I16": I16
  of "BF16": BF16
  of "I32": I32
  of "I64": I64
  else: raise newException(ValueError, "Unknown data type: " & s)

