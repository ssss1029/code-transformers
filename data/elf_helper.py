from elftools.elf.elffile import ELFFile
from elftools.elf.sections import SymbolTableSection


def extract_functions_from_symbol_table(binary):
    '''
    Extracts functions from the binary's symbol table.
    Returns a list of (function name, function address, function size).
    :param binary: ELFFile
    :return: list
    '''

    for section in binary.iter_sections():
        if (section['sh_type'] != 'SHT_SYMTAB'
                    or section['sh_entsize'] == 0):
            continue

        result = []
        for symbol in section.iter_symbols():
            if (symbol['st_info']['type'] == 'STT_FUNC' and
                    symbol['st_size'] > 0):
                result.append((symbol.name, symbol['st_value'], symbol['st_size']))
        return result
    return []

def get_text(binary):
    '''
    :param binary: ELFFile
    :return: (bytes, int)
    '''

    text = binary.get_section_by_name('.text')
    if text is None:
        raise RuntimeError('.text not found')
    return text.data(), text['sh_addr']

def open_binary(binary_path):
    try:
        return ELFFile(open(binary_path, 'rb'))
    except Exception as e:
        return None