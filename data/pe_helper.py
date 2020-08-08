import os
import pefile

def extract_functions_from_symbol_table(binary):
  '''
  Extracts functions from the binary's symbol table.
  Returns a list of (function name, function address, function size).
  :param binary: pefile.PE
  :return: list
  '''
  result = []

  # We don't know how to use the built-in symbol table
  assert binary.FILE_HEADER.PointerToSymbolTable == 0

  # Parse the ground truth file
  binary_prefix, binary_name = os.path.split(binary.name)
  gt_file = os.path.join(binary_prefix, '../gt/function/', binary_name)
  gt_f = open(gt_file, 'r')
  for line in gt_f:
    elements = line.split()
    assert len(elements) == 2
    func_begin = int(elements[0], 16)
    func_end = int(elements[1], 16)
    func_length = func_end - func_begin
    if func_length > 0:
      result.append(('<UNKNOWN>', func_begin, func_length))

  return result

def _remove_nul(str):
  return ''.join(c for c in str if c != '\0')

def get_text(binary):
  '''
  :param binary: pefile.PE
  :return: (bytes, int)
  '''
  for section in binary.sections:
    if _remove_nul(section.Name) != '.text':
      continue
    return section.get_data(), section.VirtualAddress + binary.OPTIONAL_HEADER.ImageBase

  raise RuntimeError('.text not found')

def open(binary_path):
    try:
        binary = pefile.PE(binary_path, fast_load=True)
    except Exception as e:
        return None
    binary.name = binary_path
    return binary
