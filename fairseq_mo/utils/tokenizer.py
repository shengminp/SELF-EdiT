import os
import re
import collections
from selfies import encoder
from selfies.constants import INDEX_CODE
from typing import List, Optional, Tuple
from transformers import PreTrainedTokenizer

MAX_RING_COUNTS = 20

def parse_key(x):
    if x[0] == '%':
        return int(x[1:])
    else:
        return int(x)
    
def reduce_ring_number(smiles):
    ## create a generator
    chars_iter = (c for c in smiles)
    
    ## init output
    chars = []
    
    ## init
    mapping = {}
    rids_not_used = [f'{i}' for i in range(1,10)] + [f'%{i}' for i in range(10,MAX_RING_COUNTS)]
    
    ## get an initial character
    c = next(chars_iter, None)
    
    ## do
    while c is not None:
        ## exception of %
        if c == '%':
            c += next(chars_iter) + next(chars_iter) 
        ## Case1: ion symbols
        if c == '[':
            while c != ']':
                chars.append(c)
                c = next(chars_iter)
            assert c == ']'
            chars.append(c)
        ## Case2: ring
        elif c.isdigit() or ('%' in c):
            ## close with a new ring number
            if c in mapping:
                c_new = mapping.pop(c)
                rids_not_used.append(c_new)
                rids_not_used = list(sorted(rids_not_used, key=parse_key))
            ## open with a new ring number
            else:
                c_new = rids_not_used.pop(0)
                mapping[c] = c_new
            chars.append(c_new)
        ## Case3: other symbols
        else:
            chars.append(c)
        ## next character
        c = next(chars_iter, None)
        
    ## error check
    assert len(mapping) == 0
    
    return ''.join(chars)
    
def find_all_not_used_rids(smiles):
    ## create a generator
    chars_iter = (c for c in smiles)
    
    ## init output
    rids_not_used = [f'{i}' for i in range(1,10)] + [f'%{i}' for i in range(10,MAX_RING_COUNTS)]
    
    ## get an initial character
    c = next(chars_iter, None)
    
    ## search all not used numbers
    while c is not None:
        ## exception of %
        if c == '%':
            c += next(chars_iter) + next(chars_iter) 
        ## Case1: ion symbols
        if c == '[':
            while c != ']':
                c = next(chars_iter)
            assert c == ']'
        ## Case2: ring
        elif c in rids_not_used:
            _ = rids_not_used.pop(rids_not_used.index(c))
        ## next character
        c = next(chars_iter, None)
    
    return rids_not_used

def rearrange_ring_number(smiles):
    ## create a generator
    chars_iter = (c for c in smiles)
    
    ## init output
    chars = []
    
    ## init
    mapping = {}
    rids = [f'{i}' for i in range(1,10)] + [f'%{i}' for i in range(10,MAX_RING_COUNTS)]
    rid2stat = {idx:0 for idx in rids} # 0: not_used, 1: open, 2: closed
    
    ## search all not used numbers
    rids_not_used = find_all_not_used_rids(smiles)
    
    ## get an initial character
    c = next(chars_iter, None)
    
    ## do arrangement
    while c is not None:
        ## exception of %
        if c == '%':
            c += next(chars_iter) + next(chars_iter) 
        ## Case1: ion symbols
        if c == '[':
            while c != ']':
                chars.append(c)
                c = next(chars_iter)
            assert c == ']'
            chars.append(c)
        ## Case2: other symbols
        else:
            status = rid2stat.get(c, -1)
            ## ring open
            if status == 0:
                rid2stat[c] = 1
            ## ring close
            elif status == 1:
                rid2stat[c] = 2
            ## ring id rearrangement
            elif status == 2:
                ## close
                if c in mapping:
                    c_new = mapping.pop(c)
                    assert rid2stat[c_new] == 1
                    rid2stat[c_new] = 2
                ## open
                else:
                    c_new = rids_not_used.pop(0)
                    mapping[c] = c_new
                    assert rid2stat[c_new] == 0
                    rid2stat[c_new] = 1
                c = c_new
            chars.append(c)
        ## next character
        c = next(chars_iter, None)          
    
    ## error check
    assert len(mapping) == 0
    
    return ''.join(chars)

def change_tetrahedral_carbon(smiles):
    smiles = smiles.replace('[C@@H]', 'tmp').replace('[C@H]', 'tmp2').replace('[CH]', 'tmp3')
    smiles = smiles.replace('tmp', '[C@H]').replace('tmp2', '[CH]').replace('tmp3', '[C@@H]')
    
    smiles = smiles.replace('[C@@]', 'tmp').replace('[C@]', 'tmp2').replace('[C]', 'tmp3')
    smiles = smiles.replace('tmp', '[C@]').replace('tmp2', '[C]').replace('tmp3', '[C@@]')
    return smiles

def branch_based_standardization(smiles, use_rearrange_ring_number=True, use_reduce_ring_number=True, verbose=0):
    ## Rearrange ring numbers in smiles
    if use_rearrange_ring_number:
        smiles = rearrange_ring_number(smiles)
        
    ## Decompose the given SMILES string into 3 substrings (source, branch1, branch2)
    substrings = ['', '', '']
    MAX_PNUM = 2 # len(substrings) - 1
    
    snum = 0 # stack height
    pnum = 0 # partition number
    pnum_prev = 0
    ion_flag = False
    
    for i, symbol in enumerate(smiles):
        
        if verbose > 0:
            print(i, symbol, pnum, snum, MAX_PNUM)
        
        if symbol == '[':
            assert not ion_flag
            ion_flag = True
            substrings[pnum] += symbol
        elif symbol == ']':
            assert ion_flag
            ion_flag = False
            substrings[pnum] += symbol
        elif ion_flag:
            substrings[pnum] += symbol
        else:
            if symbol == '(':
                snum += 1
                if snum == 1:
                    if smiles[i-1] == ')' and pnum == 2 and pnum_prev == 1:
                        pnum_prev = pnum
                        pnum = min(pnum+1, MAX_PNUM)
                        substrings.append('')
                        MAX_PNUM = 3
                    else:
                        pnum_prev = pnum
                        pnum = min(pnum+1, MAX_PNUM)
                substrings[pnum] += symbol
            elif symbol == ')':
                substrings[pnum] += symbol
                snum -= 1
                if snum == 0:
                    pnum_prev = pnum
                    pnum = min(pnum+1, MAX_PNUM)
            else:
                substrings[pnum] += symbol            
            
    ## stack check
    assert snum == 0
    assert not ion_flag
    
    if verbose > 0:
        print(substrings)
    
    ## branch exists?
    if len(substrings[1]) == 0:
        return substrings[0]
    elif len(substrings[2]) == 0:
        return substrings[0] + substrings[1]
    
    if len(substrings) == 3:
        ## remove the outermost bracket
        substrings[1] = substrings[1][1:-1]
        ## sort by length
        if len(substrings[1]) > len(substrings[2]):
            substrings[1], substrings[2] = substrings[2], substrings[1]
            substrings[0] = change_tetrahedral_carbon(substrings[0])
        ## assemble
        reconstructed = substrings[0]
        reconstructed += '(' + branch_based_standardization(substrings[1], False, False, verbose) + ')'
        reconstructed += branch_based_standardization(substrings[2], False, False, verbose)
    elif len(substrings) == 4:
        ## remove the outermost bracket
        substrings[1] = substrings[1][1:-1]
        substrings[2] = substrings[2][1:-1]
        ## sort by length
        if len(substrings[1]) > len(substrings[2]):
            substrings[1], substrings[2] = substrings[2], substrings[1]
            substrings[0] = change_tetrahedral_carbon(substrings[0])
        if len(substrings[2]) > len(substrings[3]):
            substrings[2], substrings[3] = substrings[3], substrings[2]
            substrings[0] = change_tetrahedral_carbon(substrings[0])
            substrings[1] = change_tetrahedral_carbon(substrings[1])
        if len(substrings[1]) > len(substrings[2]):
            substrings[1], substrings[2] = substrings[2], substrings[1]
            substrings[0] = change_tetrahedral_carbon(substrings[0])
        ## assemble
        reconstructed = substrings[0]
        reconstructed += '(' + branch_based_standardization(substrings[1], False, False, verbose) + ')'
        reconstructed += '(' + branch_based_standardization(substrings[2], False, False, verbose) + ')'
        reconstructed += branch_based_standardization(substrings[3], False, False, verbose)
        
    if use_reduce_ring_number:
        reconstructed = reduce_ring_number(reconstructed)
        
    ## rdkit SMILES parse error
    #assert '((' not in reconstructed and '))' not in reconstructed

    return reconstructed

def get_index_from_selfies(*symbols: List[str]) -> int:
    index = 0
    for i, c in enumerate(reversed(*symbols)):
        index += INDEX_CODE.get(c, 0) * (len(INDEX_CODE) ** i)
    return index

ATOM_LIST = ['[C]','[O]','[N]','[=N]','[=C]','[#C]','[S]','[P]']

def add_token(insert_list, token_list, count, count_size):
    add_sign=True
    result=None
    remain=None
    
    for i, token in enumerate(reversed(token_list)):
        if '[Ring' in token:
            insert_list.insert(0, token)
        elif 'Branch' in token:
            # in case of [Ring][Branch]
            if i+1 != len(token_list) and '[Ring' in token_list[len(token_list)-i-2]:
                insert_list.insert(0, token)
            # in case of [Branch][Branch]
            elif i+1 != len(token_list) and 'Branch' in token_list[len(token_list)-i-2]:
                insert_list.insert(0, token)
            # in case of [Branch][Atom]
            elif insert_list[0] in ATOM_LIST:
                insert_list.insert(0, token)
                count = count-1
            else:
                insert_list.insert(0, token)
        else:
            insert_list.insert(0, token)
            count += 1
        if count == count_size:
            result = ''.join(insert_list)
            count=0
            count_size=0
            add_sign=False
            if i+1 != len(token_list):
                remain = ''.join(token_list[:len(token_list)-i-1])
            break
    return insert_list, count, count_size, add_sign, (result, remain)

def merge_ring(sel_frag):
    sign = '[Ring'
    add = False
    c = 0
    count_size = 0
    insert = None
    new_frag = list()
    
    for token in reversed(sel_frag):
        splited_token = re.split(r'(?=\[)(?<=\])', token)
        if sign in token:
            size = token.split(sign)[1][0]
            sign_idx = splited_token.index(sign+size+']')

            # the case of [Branch][Ring]
            if "Branch" in splited_token[sign_idx-1] and add is False:
                new_frag.append(token)
                continue
            elif c < count_size and add is True:
                insert, c, count_size, add, result = add_token(insert, splited_token, c, count_size)
                result, remain = result
                if result is not None:
                    new_frag.append(result)
                if remain is not None:
                    new_frag.append(remain)
            else:
                count_symbol = [s for s in splited_token[sign_idx+1: sign_idx+1+int(size)]]
                count_size = get_index_from_selfies(count_symbol) + 2

                insert = splited_token[sign_idx:]
                add=True
                c = 0

                insert, c, count_size, add, result = add_token(insert, splited_token[0:sign_idx], c, count_size)
                result, remain = result
                if result is not None:
                    new_frag.append(result)
                if remain is not None:
                    new_frag.append(remain)

        elif add is True:
            insert, c, count_size, add, result = add_token(insert, splited_token, c, count_size)
            result, remain = result
            if result is not None:
                new_frag.append(result)
            if remain is not None:
                new_frag.append(remain)
        else:
            new_frag.append(token)
            
    return [frag for frag in reversed(new_frag)]

def selfies_tokenizer(selfies):
    import selfies as sf
    
    fragments = []
    pos = 0
    symbols = list(sf.split_selfies(selfies))
    L = len(symbols)
    b = ''
    while pos < L:
        v = symbols[pos]
        ## Case 1: Branch symbol (e.g. v = '[Branch1]')
        if 'ch' == v[-4:-2]:
            ## save
            if len(b) > 0: fragments.append(b)
            ## branch size (Q)
            n = int(v[-2])
            Q = 1 + sf.grammar_rules.get_index_from_selfies(*symbols[pos + 1:pos + 1 + n])
            ## branch
            b = ''.join(symbols[pos:pos + 1 + n + Q])
            ## save and reset
            fragments.append(b)
            b = ''
        ## Case 2: Ring symbol (e.g. v = '[Ring2]')
        elif 'ng' == v[-4:-2]:
            ## number of symbols for ring size (n)
            n = int(v[-2])
            Q = 0
            ## branch
            b += ''.join(symbols[pos:pos + 1 + n + Q])
            ## save and reset
            fragments.append(b)
            b = ''
        #elif pos == L-1:
            #fragments.append(v)
        else:
            b += v
            n = 0
            Q = 0
        ## update pos
        pos += 1 + n + Q
        if pos == L and b is not '':
            fragments.append(b)

    fragments = merge_ring(fragments)
    return fragments

def atomwise_tokenizer(smiles):
        """
        Tokenize a SMILES molecule at atom-level:
            (1) 'Br' and 'Cl' are two-character tokens
            (2) Symbols with bracket are considered as tokens
            (3) All other symbols are tokenized on character level.
        
        """
        import re
        from functools import reduce
        regex = '(\[[^\[\]]{1,10}\])'
        char_list = re.split(regex, smiles)
        tokens = []                
            
        for char in char_list:
            if char.startswith('['):
                tokens.append(str(char))
            else:
                chars = [unit for unit in char]
                [tokens.append(i) for i in chars]
                    
        #fix the 'Br' be splited into 'B' and 'r'
        if 'r' in tokens:
            for index, tok in enumerate(tokens):
                if tok == 'r':
                    if tokens[index-1] == 'B':
                            tokens[index-1: index+1] = [reduce(lambda i, j: i + j, tokens[index-1 : index+1])]
            
        #fix the 'Cl' be splited into 'C' and 'l'
        if 'l' in tokens:
            for index, tok in enumerate(tokens):
                if tok == 'l':
                    if tokens[index-1] == 'C':
                            tokens[index-1: index+1] = [reduce(lambda i, j: i + j, tokens[index-1 : index+1])]
        return tokens

def s2s_tokenize(smiles):
    smiles = smiles.rstrip('\n')
    selfies = encoder(smiles)
    return selfies_tokenizer(selfies)

def s2c_tokenize(smiles):
    smiles = smiles.rstrip('\n')
    selfies = encoder(smiles)
    return atomwise_tokenizer(selfies)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

def load_vocab(vocab_file):
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab

class MoTokenizer(PreTrainedTokenizer):
    vocab_files_names = VOCAB_FILES_NAMES
    
    def __init__(
        self,
        vocab_file,
        do_lower_case=False,
        do_basic_tokenize=False,
        never_split=None,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        tokenize_chinese_chars=True,
        strip_accents=None,
        **kwargs
    ):
        super().__init__(
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            never_split=never_split,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            **kwargs,
        )

        if not os.path.isfile(vocab_file):
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained"
                " model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        
    def _atomwise_tokenizer(self, smiles):
        """
        Tokenize a SMILES molecule at atom-level:
            (1) 'Br' and 'Cl' are two-character tokens
            (2) Symbols with bracket are considered as tokens
            (3) All other symbols are tokenized on character level.
        
        """
        import re
        from functools import reduce
        regex = '(\[[^\[\]]{1,10}\])'
        char_list = re.split(regex, smiles)
        tokens = []                
            
        for char in char_list:
            if char.startswith('['):
                tokens.append(str(char))
            else:
                chars = [unit for unit in char]
                [tokens.append(i) for i in chars]
                    
        #fix the 'Br' be splited into 'B' and 'r'
        if 'r' in tokens:
            for index, tok in enumerate(tokens):
                if tok == 'r':
                    if tokens[index-1] == 'B':
                            tokens[index-1: index+1] = [reduce(lambda i, j: i + j, tokens[index-1 : index+1])]
            
        #fix the 'Cl' be splited into 'C' and 'l'
        if 'l' in tokens:
            for index, tok in enumerate(tokens):
                if tok == 'l':
                    if tokens[index-1] == 'C':
                            tokens[index-1: index+1] = [reduce(lambda i, j: i + j, tokens[index-1 : index+1])]
        return tokens
    
    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab, **self.added_tokens_encoder)
    
    def _tokenize(self, text):
        split_tokens = self._atomwise_tokenizer(text)
        return split_tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)
    
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        out_string = " ".join(tokens).strip()
        return out_string
    
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A BERT sequence has the following format:
        - single sequence: `[CLS] X [SEP]`
        - pair of sequences: `[CLS] A [SEP] B [SEP]`
        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.
        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.
        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A BERT sequence
        pair mask has the following format:
        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```
        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).
        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        index = 0
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
            )
        else:
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    index = token_index
                writer.write(token + "\n")
                index += 1
        return (vocab_file,)