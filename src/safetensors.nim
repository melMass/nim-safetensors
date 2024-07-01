# NOTE: basic implementation of safetensors
# spec here: https://github.com/huggingface/safetensors
import strformat, json, sequtils, tables, streams, endians, os

import results
# import chronicles

import safetensors/types
import safetensors/error

proc newSafetensor*(metadata: Table[string, string] = initTable[string, string]()): Safetensor =
  Safetensor(metadata: metadata, tensors: initTable[string, TensorHeader](),
      data: @[])

proc addTensor*[T](st: var Safetensor, name: string, data: openArray[T],
    shape: openArray[int]) =
  let startOffset = st.data.len
  let dataSize = data.len * sizeof(T)
  st.data.setLen(startOffset + dataSize)
  copyMem(st.data[startOffset].addr, data[0].unsafeAddr, dataSize)

  st.tensors[name] = TensorHeader(
    dtype: when T is float32: F32 elif T is float64: F64 else: I32, # TODO: add other case
    shape: @shape,
    data_offsets: [startOffset, startOffset + dataSize]
  )

proc keys*(s: Safetensor): seq[string] {.noSideEffect.} =
  toSeq(s.tensors.keys)

proc loadSafetensor*(filename: string): SResult[Safetensor] =
  # debug "Loading stream"

  # var file: File
  #
  # if not open(file, filename):
  #   result.err SafeTensorError(kind: IoError, message: "Unable to open file: " & filename)
  #   return
  #
  # defer: file.close()
  #
  var fileStream = newFileStream(filename)
  if fileStream == nil:
    result.err SafeTensorError(kind: IoError,
        message: fmt"Unable to open file stream: {filename}")
    return

  defer: fileStream.close()

  var headerSize: uint64
  if fileStream.readData(headerSize.addr, sizeof(uint64)) != sizeof(uint64):
    result.err SafeTensorError(kind: IoError,
       message: "Failed to read header size")
    return

  littleEndian64(headerSize.addr, headerSize.addr)

  if headerSize > 100_000_000:
    result.err SafeTensorError(kind: HeaderTooLarge,
        message: fmt"Header is {headerSize}")
    return

  # debug fmt"Header size: {headerSize}"

  let headerStr = fileStream.readStr(headerSize.int)
  let headerJson = parseJson(headerStr)

  var st = Safetensor()
  st.metadata = initTable[string, string]()
  st.tensors = initTable[string, TensorHeader]()

  if "__metadata__" in headerJson:
    for key, value in headerJson["__metadata__"].getFields:
      st.metadata[key] = value.getStr

  for key, value in headerJson.getFields:
    if key != "__metadata__":
      var tensorHeader = TensorHeader(
        dtype: value["dtype"].getStr.toDataType(),
        shape: value["shape"].getElems.mapIt(it.getInt),
        # data_offsets: value["data_offsets"].getElems.mapIt(it.getInt)
        data_offsets: (start: value["data_offsets"][0].getInt, `end`: value[
            "data_offsets"][0].getInt)
      )
      st.tensors[key] = tensorHeader


  # read the remaining data
  # let dataSize = getFileSize(file) - fileStream.getPosition
  st.data = newSeq[byte]()
  while not fileStream.atEnd():
    st.data.add fileStream.readUint8()
  # discard fileStream.readData(st.data[0].addr, dataSize)

  ok(st)


proc save*(st: Safetensor, filename: string, force = false): SResult[void] =

  if fileExists(filename) and not force:
    result.err SafeTensorError(kind: FileExists, message: filename & ", use 'force:true' to overwrite")
    return

  var headerJson = newJObject()

  if st.metadata.len > 0:
    var metadataJson = newJObject()
    for key, value in st.metadata:
      metadataJson[key] = newJString(value)
    headerJson["__metadata__"] = metadataJson

  for tensorName, tensorHeader in st.tensors:
    var tensorJson = newJObject()
    tensorJson["dtype"] = newJString($tensorHeader.dtype)
    tensorJson["shape"] = newJArray()
    for dim in tensorHeader.shape:
      tensorJson["shape"].add(newJInt(dim))
    tensorJson["data_offsets"] = newJArray()
    tensorJson["data_offsets"].add(newJInt(tensorHeader.data_offsets[0]))
    tensorJson["data_offsets"].add(newJInt(tensorHeader.data_offsets[1]))
    headerJson[tensorName] = tensorJson

  let headerStr = $headerJson
  var headerSize: uint64 = headerStr.len.uint64
  var headerSizeLe: uint64

  littleEndian64(headerSizeLe.addr, headerSize.addr)

  var fileStream = newFileStream(filename, fmWrite)
  if fileStream == nil:
    raise newException(system.IOError, "Unable to create file: " & filename)

  defer: fileStream.close()

  fileStream.write(headerSizeLe)
  fileStream.write(headerStr)
  fileStream.writeData(st.data[0].addr, st.data.len)


proc getTensor*[T](st: Safetensor, name: string): SResult[seq[T]] =
  ## # Usage:
  ## ```nim
  ## let floatTensor = st.getTensor[float32]("my_tensor")
  ## ```

  let rawData = st.getTensorData(name)
  result = newSeq[T](rawData.len div sizeof(T))
  copyMem(result[0].addr, rawData[0].addr, rawData.len)

proc getMetadata*(st: Safetensor, key: string, default: string = ""): string =
  st.metadata.getOrDefault(key, default)

proc getTensorData*(safetensor: Safetensor, tensorName: string): seq[byte] =
  if tensorName notin safetensor.tensors:
    raise newException(KeyError, &"Tensor '{tensorName}' not found in safetensor")

  let tensor = safetensor.tensors[tensorName]
  let startOffset = tensor.data_offsets[0]
  let endOffset = tensor.data_offsets[1]
  safetensor.data[startOffset ..< endOffset]

iterator tensors*(st: Safetensor): tuple[name: string, header: TensorHeader,
    data: seq[byte]] =
  for name, header in st.tensors:
    yield (name, header, st.getTensorData(name))

proc checkShape*(st: Safetensor, name: string, expectedShape: openArray[int]): bool =
  ## # Usage:
  ## ```nim
  ## if st.checkShape("embedding", [100, 768]):
  ##   echo "Embedding layer has correct shape"
  ## ```
  name in st.tensors and st.tensors[name].shape == @expectedShape

when(isMainModule):
  # Example usage
  echo "Loading safetensor file..."
  let safetensor = loadSafetensor("G:/MODELS/LoRa/smooth_lora.safetensors")
  if safetensor.isErr():
    safetensor.error.stdOut()
    quit 1

  echo "Loaded"
  let st = safetensor.get()

  if st.metadata.len == 0:
    echo "No metadata found"
  else:
    echo st.metadata

  echo fmt"Found {st.keys().len} tensors"
  echo "Saving..."
  let res = st.save("./out_lora.safetensors")

  if res.isErr():
    res.error.stdOut()
    quit 1

  echo "Saved!"

  let sf2 = loadSafetensor("./out_lora.safetensors")
  if sf2.isErr():
    safetensor.error.stdOut()
    quit 1

  echo "Loaded back!"

  let st2 = sf2.get()
  if st2.metadata.len == 0:
    echo "No metadata found"
  else:
    echo st2.metadata

  echo fmt"Found {st2.keys().len} tensors"

