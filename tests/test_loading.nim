import unittest
import results
import safetensors

test "Loading a basic LoRa":
  let safetensor = loadSafetensor("G:/MODELS/LoRa/smooth_lora.safetensors")

  check safetensor.isOk == true
  let st = safetensor.get()
  check st.keys().len == 256
