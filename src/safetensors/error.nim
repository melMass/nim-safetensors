import strutils, json, strformat
import results
import ./types

type
  SafeTensorErrorKind* = enum
    InvalidHeader, InvalidHeaderStart, InvalidHeaderDeserialization,
      HeaderTooLarge, FileExists,
    HeaderTooSmall, InvalidHeaderLength, TensorNotFound, TensorInvalidInfo,
    InvalidOffset, IoError, JsonError, InvalidTensorView,
      MetadataIncompleteBuffer,
    ValidationOverflow

  SafeTensorError* = ref object of CatchableError
    case kind*: SafeTensorErrorKind
      of InvalidHeader: discard
      of InvalidHeaderStart: discard
      of InvalidHeaderDeserialization: discard
      of InvalidHeaderLength: discard
      of [TensorNotFound, InvalidOffset]: tensor_name*: string
      of TensorInvalidInfo: discard
      of JsonError: jsonError*: ref JsonParsingError
      of InvalidTensorView:
        dtype*: DataType
        shape*: seq[int]
        size*: int
      of MetadataIncompleteBuffer: discard
      of ValidationOverflow: discard
      # of [HeaderTooLarge, HeaderTooSmall, IoError, FileExists ]: message*: string
      else:
        message*: string

  SResult*[T] = Result[T, SafeTensorError]


proc log_red(ss: varargs[string, `$`]) =
  echo "\e[31m" & @ss.join(" ") & "\e[0m"

proc stdOut*(self: SafeTensorError) =
  case self.kind:
    of InvalidHeader: log_red "Invalid Header"
    of InvalidHeaderStart: log_red "Invalid Header Start"
    of InvalidHeaderDeserialization: log_red "Invalid Header Deserialization"
    of HeaderTooLarge: log_red fmt"Header too large: {self.message}"
    of [HeaderTooSmall]: log_red fmt"Header too small: {self.message}"
    of [IoError]: log_red fmt"IO Error: {self.message}"
    of InvalidHeaderLength: log_red "Invalid Header Length"
    of [TensorNotFound]: log_red fmt"Tensor '{self.tensorName}' not found in safetensor"
    of [InvalidOffset]: log_red fmt"Invalid offset for tensor: '{self.tensor_name}'"
    of TensorInvalidInfo: log_red "Invalid tensor info"
    of JsonError: log_red self.jsonError.msg
    of InvalidTensorView: log_red fmt"Invalid tensor view for dtype: {self.dtype}, shape: {self.shape}, size: {self.size}"
    of MetadataIncompleteBuffer: log_red "Metadata incomplete buffer"
    of ValidationOverflow: log_red "Validation overflow"
    of FileExists: log_red fmt"File Exists: {self.message}"
    else: log_red "Something went wrong..."


